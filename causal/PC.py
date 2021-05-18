import numpy as np
import pandas as pd
import networkx as nx

# from gsq.ci_tests import ci_test_bin, ci_test_dis
# from gsq.gsq_testdata import bin_data, dis_data
# from pycit import itest
# from pycit import citest

from scipy.stats import chi2_contingency
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

from itertools import combinations, permutations
import logging
_logger = logging.getLogger(__name__)



class PC:
    
    is_var_discrete_map = {};
    distinct_val_ratio = 0.05;
    
    def __init__(self, data_df):
        is_var_discrete_map = {};
        for var in data_df.columns:
            self.is_var_discrete_map[var] = 1.*data_df[var].nunique()/data_df[var].count() < self.distinct_val_ratio;
    
    knn_num = 5;
    #knn_num = round(dataset.shape[0]**0.5)
    n_jobs = 4;
    k_perm = 20;
    distance_matric = 'minkowski';
    gpdc = GPDC(significance='analytic', gp_params=None);

    def cdi_test(self,data,x, y, z_set):
    
        #Do independence test (conditional or not) according to variable data types
        x_values = data[x].values;
        y_values = data[y].values;
    
        #print(z_values.shape);
        is_x_dis = self.is_var_discrete_map[x];
        is_y_dis = self.is_var_discrete_map[y];
        pval = 0;
        
        if len(z_set)<=0:
            #Independence test
            #If all the variables are discrete
            if is_x_dis and is_y_dis:
                obs = pd.crosstab(x_values,y_values);
                g,pval,dof,expctd = chi2_contingency(obs, lambda_="log-likelihood");#
                #pval = itest(x_values, y_values, statistic="mixed_mi",statistic_args={'k': 20}, test_args={'n_jobs': 4});
            #If all the variables are continuous
            elif (not is_x_dis) and (not is_y_dis):
                x = x_values.reshape(x_values.shape[0], 1);
                y = y_values.reshape(y_values.shape[0], 1);
                val, pval = self.gpdc.run_test_raw(x,y,z=None);
                #pval = itest(x_values, y_values,statistic_args={'k': knn_num}#bi_ksg_mi , statistic="bi_ksg_mi"
                #             , test_args={'statistic': 'bi_ksg_mi','n_jobs': n_jobs,'k_perm':k_perm,'metric':distance_matric});#bi_ksg_mi
                #print("i test: data type->{} pvalue->{}".format("it-continuous",pval));
            #If the variables are mixed
            else:
                x = x_values.reshape(x_values.shape[0], 1);
                y = y_values.reshape(y_values.shape[0], 1);
                val, pval = self.gpdc.run_test_raw(x,y,z=None);
                #pval = itest(x_values, y_values,statistic_args={'k': knn_num}#,statistic="mixed_mi"
                #             , test_args={'statistic': 'mixed_mi','n_jobs': n_jobs,'k_perm':k_perm,'metric':distance_matric});
                #print("i test: data type->{} pvalue->{}".format("it-mixed",pval));
        else:
            #Conditional independence test
            z_values = data[list(z_set)].values;
        
            z_has_dis = False;
            z_has_cont = False;
            for z_ele in z_set:
                if self.is_var_discrete_map[z_ele]:
                    z_has_dis = True;
                else:
                    z_has_cont = True;
                    
           #If all the variables are discrete
            if is_x_dis and is_x_dis and not z_has_cont:
                pval = citest(x_values, y_values, z_values,statistic_args={'k': self.knn_num}#, statistic="mixed_cmi"
                          ,test_args={'statistic':'mixed_cmi','n_jobs': n_jobs,'k_perm':self.k_perm,'metric':self.distance_matric});
                #print("ci test: data type->{} pvalue->{}".format("cit-discrete",pval));
            #If all the variables are continuous
            elif (not is_x_dis) and (not is_x_dis) and (not z_has_dis):
                x = x_values.reshape(x_values.shape[0], 1);
                y = y_values.reshape(y_values.shape[0], 1);
                z = z_values.reshape(z_values.shape[0], len(z_set));
                val, pval = self.gpdc.run_test_raw(x,y,z=z);
                #pval = citest(x_values, y_values, z_values,statistic_args={'k': knn_num}#, statistic="bi_ksg_mi"
                #              ,test_args={'statistic':'bi_ksg_mi','n_jobs': n_jobs,'k_perm':k_perm,'metric':distance_matric});
                #print("ci test: data type->{} pvalue->{}".format("cit-continuous",pval));
            #If the variables are mixed, use mutual information
            else:
                x = x_values.reshape(x_values.shape[0], 1);
                y = y_values.reshape(y_values.shape[0], 1);
                z = z_values.reshape(z_values.shape[0], len(z_set));
                val, pval = self.gpdc.run_test_raw(x,y,z=z);
                #pval = citest(x_values, y_values, z_values,statistic_args={'k': knn_num}#,statistic="mixed_cmi"
                #              ,test_args={'statistic':'mixed_cmi','n_jobs':n_jobs,'k_perm':k_perm,'metric':distance_matric});
                #print("ci test: data type->{} pvalue->{}".format("cit-mixed",pval));
            
        return pval;
    
    
    def estimate_skeleton(self,data_df, g, alpha, **kwargs):
        

        def method_stable(kwargs):
            return ('method' in kwargs) and kwargs['method'] == "stable";

        node_ids = data_df.columns;
        node_size = data_df.shape[1];
        sep_set = [[set() for i in range(node_size)] for j in range(node_size)];
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None;
                sep_set[j][i] = None;

        l = 0
        while True:
            cont = False
            remove_edges = []
            for (i, j) in permutations(node_ids, 2):
                adj_i = list(g.neighbors(i))
                if j not in adj_i:
                    continue
                else:
                    adj_i.remove(j)
                if len(adj_i) >= l:
                    _logger.debug('testing %s and %s' % (i,j))
                    _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
                    if len(adj_i) < l:
                        continue
                    for k in combinations(adj_i, l):
                        _logger.debug('indep prob of %s and %s with subset %s'% (i, j, str(k)))
                        
                        
                        #p_val = indep_test_func(data_matrix, i, j, set(k),**kwargs);
                        #p_val = ci_test_dis(data_df.values, i, j, set(k),**kwargs);
                        
                        #p_val = ci_test_dis(data_df.values, i, j, set(k),**kwargs);#,levels=levels
                        p_val = self.cdi_test(data=data_df,x=i,y=j,z_set=set(k));
                        _logger.debug('p_val is %s' % str(p_val))
                        if p_val > alpha:
                            if g.has_edge(i, j):
                                _logger.debug('p: remove edge (%s, %s)' % (i, j))
                                if method_stable(kwargs):
                                    remove_edges.append((i, j))
                                else:
                                    g.remove_edge(i, j)
                            sep_set[i][j] |= set(k)
                            sep_set[j][i] |= set(k)
                            break
                    cont = True
            l += 1
            if method_stable(kwargs):
                g.remove_edges_from(remove_edges)
            if cont is False:
                break
            if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
                break

        return (g, sep_set)
        
    
    def estimate_cpdag(self,skel_graph, sep_set):
        """Estimate a CPDAG from the skeleton graph and separation sets
        returned by the estimate_skeleton() function.

        Args:
            skel_graph: A skeleton graph (an undirected networkx.Graph).
            sep_set: An 2D-array of separation set.
                The contents look like something like below.
                    sep_set[i][j] = set([k, l, m])

        Returns:
            An estimated DAG.
        """
        dag = skel_graph.to_directed()
        node_ids = skel_graph.nodes()
        for (i, j) in combinations(node_ids, 2):
            adj_i = set(dag.successors(i))
            if j in adj_i:
                continue
            adj_j = set(dag.successors(j))
            if i in adj_j:
                continue
            if sep_set[i][j] is None:
                continue
            common_k = adj_i & adj_j
            for k in common_k:
                if k not in sep_set[i][j]:
                    if dag.has_edge(k, i):
                        _logger.debug('S: remove edge (%s, %s)' % (k, i))
                        dag.remove_edge(k, i)
                    if dag.has_edge(k, j):
                        _logger.debug('S: remove edge (%s, %s)' % (k, j))
                        dag.remove_edge(k, j)

        def _has_both_edges(dag, i, j):
            return dag.has_edge(i, j) and dag.has_edge(j, i)

        def _has_any_edge(dag, i, j):
            return dag.has_edge(i, j) or dag.has_edge(j, i)

        def _has_one_edge(dag, i, j):
            return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
                    (not dag.has_edge(i, j)) and dag.has_edge(j, i))

        def _has_no_edge(dag, i, j):
            return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

        # For all the combination of nodes i and j, apply the following
        # rules.
        old_dag = dag.copy()
        while True:
            for (i, j) in combinations(node_ids, 2):
                # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
                # such that k and j are nonadjacent.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Look all the predecessors of i.
                    for k in dag.predecessors(i):
                        # Skip if there is an arrow i->k.
                        if dag.has_edge(i, k):
                            continue
                        # Skip if k and j are adjacent.
                        if _has_any_edge(dag, k, j):
                            continue
                        # Make i-j into i->j
                        _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                        dag.remove_edge(j, i)
                        break

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Find nodes k where k is i->k.
                    succs_i = set()
                    for k in dag.successors(i):
                        if not dag.has_edge(k, i):
                            succs_i.add(k)
                    # Find nodes j where j is k->j.
                    preds_j = set()
                    for k in dag.predecessors(j):
                        if not dag.has_edge(j, k):
                            preds_j.add(k)
                    # Check if there is any node k where i->k->j.
                    if len(succs_i & preds_j) > 0:
                        # Make i-j into i->j
                        _logger.debug('R2: remove edge (%s, %s)' % (j, i))
                        dag.remove_edge(j, i)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Find nodes k where i-k.
                    adj_i = set()
                    for k in dag.successors(i):
                        if dag.has_edge(k, i):
                            adj_i.add(k)
                    # For all the pairs of nodes in adj_i,
                    for (k, l) in combinations(adj_i, 2):
                        # Skip if k and l are adjacent.
                        if _has_any_edge(dag, k, l):
                            continue
                        # Skip if not k->j.
                        if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                            continue
                        # Skip if not l->j.
                        if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                            continue
                        # Make i-j into i->j.
                        _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                        dag.remove_edge(j, i)
                        break

                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent.
                #
                # However, this rule is not necessary when the PC-algorithm
                # is used to estimate a DAG.

            if nx.is_isomorphic(dag, old_dag):
                break
            old_dag = dag.copy()

        return dag
