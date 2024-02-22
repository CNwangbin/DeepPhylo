import biom
import pandas as pd
import numpy as np
import qiime2 as q2
from skbio import TreeNode
from qiime2.plugins.phylogeny.methods import filter_table
from deepphylo.rpca import rpca_table,rclr
from deepphylo.preprocessing import fast_unifrac,matrix_rclr
from collections import Counter
from deepphylo.plot import plot_2d,  normalize, reducer, get_evol_feature
import matplotlib.pyplot as plt

def import_and_process_data(table_path, metadata_path, tree_path):
    # import data
    table = q2.Artifact.import_data('FeatureTable[Frequency]', biom.load_table(table_path))
    metadata = q2.Metadata.load(metadata_path)
    tree = q2.Artifact.load(tree_path)
    table = filter_table(table, tree).filtered_table
    
    # Filter metadata
    mf = metadata.to_dataframe()
    mf = mf[mf.host_body_site == 'UBERON:skin of hand']
    mf = mf[mf.village != 'Puerto Almendras']
    mf = mf[mf.house_number.isin([k for k, v in mf.house_number.value_counts().items()
                                if v >= 3])]

    # keep shared IDs
    bt = table.view(biom.Table)
    shared_ = set(bt.ids()) & set(mf.index)
    bt = bt.filter(shared_)
    mf = mf.reindex(shared_)
    metadata = q2.Metadata(mf)

    # filter out zero sums
    n_features, n_samples = bt.shape
    # filter features by N samples presence
    min_feature_frequency = 10
    def frequency_filter(val, id_, md):
        return (np.sum(val > 0) / n_samples) > (min_feature_frequency / 100)
    min_feature_count = 10
    def observation_filter(val, id_, md):
            return sum(val) > min_feature_count
    bt = bt.filter(observation_filter, axis='observation')
    #bt = bt.filter(bt.ids('observation')[(bt.sum('observation') > 0)], axis='observation')
    bt = bt.filter(frequency_filter, axis='observation')
    bt = bt.filter(bt.ids()[(bt.sum('sample') > 0)])

    # re-import
    table = q2.Artifact.import_data('FeatureTable[Frequency]', bt)
        
    return table, bt, metadata, tree

def RPCA_with_abundance(bt, table_matrix_rclr):
    df_table_abundance_rclr = pd.DataFrame(table_matrix_rclr, index=list(bt.ids(axis='sample')), columns=list(bt.ids(axis='observation')))
    labels_str = [sid.split('.')[1] for sid in df_table_abundance_rclr.index]
    str_dict = {'Che':0, 'Iqu':1, 'Man':2, 'Manaus':2}
    labels_int = [str_dict[lab] for lab in labels_str]  
    color_map = {0:'#1f78b4', 1:'#a6cee3', 2:'#ff7f00'}
    colors = [color_map[label] for label in labels_int]
    feature_loading_abundance_3d, sample_loading_abundance_3d, eigvals_abundance_3d, proportion_explained_abundance_3d, distance_abundance_3d = rpca_table(df_table_abundance_rclr, 3)
    return table_matrix_rclr, df_table_abundance_rclr, sample_loading_abundance_3d, colors

def Phylo_RPCA(tree,bt):
    tntree = tree.view(TreeNode)
    tntree = tntree.shear(bt.ids('observation'))
    counts_by_node, tree_index, branch_lengths, fids, otu_ids = fast_unifrac(bt, tntree)
    table_matrix_phylo_rclr = matrix_rclr(counts_by_node, branch_lengths=branch_lengths)
    df_table_phylo_rclr = pd.DataFrame(table_matrix_phylo_rclr, index=list(bt.ids(axis='sample')), columns=fids)
    feature_loading_phyloRPCA_3d, sample_loading_phyloRPCA_3d, eigvals_phyloRPCA_3d, proportion_explained_phyloRPCA_3d, distance_phyloRPCA_3d = rpca_table(df_table_phylo_rclr, 3)
    return tntree, sample_loading_phyloRPCA_3d

def RPCA_PCA(bt, tntree, table_matrix_rclr,df_table_abundance_rclr):
    fid_seq_list = []
    for fid in bt.ids('observation'):
        fid_seq_list.append(fid)
    distance_matrix = np.zeros((len(fid_seq_list), len(fid_seq_list)))
    for node1 in tntree.tips():
        i = fid_seq_list.index(node1.name)
        for node2 in tntree.tips():
            j = fid_seq_list.index(node2.name)
            distance_matrix[i, j] = node1.distance(node2)
    otu_evol_embedding = reducer(distance_matrix, 'pca', 200)
    evol_feature = get_evol_feature(table_matrix_rclr, otu_evol_embedding)
    evol_feature_normalized = normalize(evol_feature)
    feature_loading_abundance_20d, sample_loading_abundance_20d, eigvals_abundance_20d, proportion_explained_abundance_20d, distance_abundance_20d = rpca_table(df_table_abundance_rclr, 20)
    sample_reduced = []
    for i, row in sample_loading_abundance_20d.iterrows():
        elements = [v for v in row.values]
        sample_reduced.append(elements)
    sample_abundance_feature_20d = np.array(sample_reduced) 
    sample_evol_feature_20d = reducer(evol_feature_normalized, method='pca', n_components=20)
    sample_abundance_feature_20d_normalized = normalize(sample_abundance_feature_20d)
    sample_evol_feature_20d_normalized = normalize(sample_evol_feature_20d)
    sample_merged_feature = np.concatenate([sample_abundance_feature_20d_normalized, sample_evol_feature_20d_normalized], axis=1)
    sample_merged_feature_2d = reducer(sample_merged_feature, method='pca', n_components=2)
    return sample_evol_feature_20d_normalized , sample_merged_feature_2d
