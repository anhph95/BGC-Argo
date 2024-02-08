#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,  fcluster, set_link_color_palette, linkage, optimal_leaf_ordering
from sklearn.decomposition import PCA
import random, colorsys
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.ticker as ticker

import colorsys
def generate_distinct_colors(n, cmap_name, saturation=0.5, lightness=0.5, random=False):
    cmap = plt.cm.get_cmap(cmap_name, n)
    colors = cmap(np.linspace(0, 1, n))

    # Convert RGB to HLS (vectorized operation)
    hls_colors = np.array([colorsys.rgb_to_hls(*color[:3]) for color in colors])

    # Adjust saturation and lightness (vectorized operation)
    hls_colors[:, 1] = lightness  # Adjust lightness
    hls_colors[:, 2] = saturation  # Adjust saturation

    # Convert HLS back to RGB (vectorized operation)
    rgb_colors = np.array([colorsys.hls_to_rgb(*hls) for hls in hls_colors])

    # Convert RGB to Hex (vectorized operation)
    hex_colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors]
    if random: random.shuffle(hex_colors)
    return hex_colors

def hclust(linkage_matrix, k, mycolor, labels, method='ward', metric='euclidean', show_label=False, threshold_bar=True):    
    # Get clusters
    cluster = fcluster(linkage_matrix,k,'maxclust')
    cluster = pd.Categorical(cluster)

    # Distance threshold
    distance_threshold = linkage_matrix[-(k - 1), 2]
    
    # Plot 
    set_link_color_palette(mycolor)
    plt.figure(figsize=(12,5))
    with plt.rc_context({'lines.linewidth': 2.5}):
        dendrogram(linkage_matrix, orientation='top', labels=labels, no_labels=~show_label,
                color_threshold=distance_threshold, above_threshold_color='black',truncate_mode=None)
    ax = plt.gca()
    ax.spines[:].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='y', labelsize=14,left=True)
    ax.set_ylabel('Distance', fontsize=16)
    if threshold_bar: plt.axhline(y=distance_threshold, c='red', lw=2, linestyle='dashed')
    plt.grid(False)
    plt.show()
    return(cluster)

def pca(df, n_components, var_list, cluster, mycolor, show_label=True, lim=5, pc2=2, tick_spacing=1, s=100):
    model = PCA(n_components).fit(df)
    pca_summary = pd.DataFrame({'Standard deviation': np.sqrt(model.explained_variance_),
                                'Proportion of variance': model.explained_variance_ratio_,
                                'Cumulative proportion': np.cumsum(model.explained_variance_ratio_)})
    
    # Get PCA scores
    scores = model.transform(df)
    
    # Get PCA variable coordinates
    corr = model.components_.T
    pcs = [f'PC{i+1}' for i in np.arange(np.shape(corr)[1])]
    pca_summary.index = pcs
    
    # Observation plot
    fig, [ax1, ax2] = plt.subplots(1,2,figsize=(12, 6))
    ax1.axhline(0, linestyle="dotted", color="black",zorder=1) # zorder to bring to back
    ax1.axvline(0, linestyle="dotted", color="black",zorder=1)
    ax1.scatter(scores[:, 0], scores[:, pc2-1], c=[mycolor[i] for i in cluster.codes], s=s, edgecolors='k')
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.set_xlabel(f"PC1 ({round(pca_summary['Proportion of variance'][0] * 100, 1)}%)", fontsize=16)
    ax1.set_ylabel(f"PC{pc2} ({round(pca_summary['Proportion of variance'][pc2-1] * 100, 1)}%)", fontsize=16)

    # Feature plot
    ax1.tick_params(labelsize=14, bottom=True, left=True)
    ax2.tick_params(labelsize=14, bottom=True, left=True)
    ax2.axhline(0, linestyle="dotted", color="black",zorder=1)
    ax2.axvline(0, linestyle="dotted", color="black",zorder=1)
    for x, y, label in zip(corr[:, 0], corr[:, pc2-1], var_list):
        ax2.arrow(0, 0, x, y, head_width=0.05, color="red", lw=2)
        if show_label: ax2.text(x*1.2, y*1.2, label, fontsize=14, color="black",va='top',ha='left')

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel(f"PC1 ({round(pca_summary['Proportion of variance'][0] * 100, 1)}%)", fontsize=16)
    ax2.set_ylabel(f"PC{pc2} ({round(pca_summary['Proportion of variance'][pc2-1] * 100, 1)}%)", fontsize=16)
    ax2.set_xticks(np.arange(-1, 1.5, 0.5))
    ax2.set_yticks(np.arange(-1, 1.5, 0.5))

    # Adjust plot
    fig.subplots_adjust(wspace=0.5)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.grid(False)
    ax2.grid(False)
    ax1.spines[:].set_linewidth(1)
    ax1.spines[:].set_color('k')
    ax2.spines[:].set_linewidth(1)
    ax2.spines[:].set_color('k')
    plt.show()
    return(pca_summary)

def hclust_crit(df, linkage_matrix, k, kmax=20):
    # Silhouette score
    sh_scores = []
    for num_clust in range(2,kmax+1):
        sh_scores.append(silhouette_score(df, fcluster(linkage_matrix, t=num_clust, criterion='maxclust')))
    # Plot silhouette
    fig, [ax1, ax2] = plt.subplots(2,1,figsize=(12, 8),sharex=True)
    ax1.plot(range(2,kmax+1),sh_scores,'o-')
    ax1.axvline(x=k, c='red', lw=2, linestyle='dashed')
    ax1.set_ylabel('Silhouette score')
    ax1.tick_params(bottom=True, left=True)
    ax1.spines[:].set_linewidth(1)
    ax1.spines[:].set_color('k')

    # Linkage distances
    linkage_distances = []
    # Determine the optimal number of clusters
    for num_clust in range(2, kmax + 1):
        linkage_distance = linkage_matrix[-num_clust, 2]
        linkage_distances.append(linkage_distance)

    # Plot the linkage distances
    ax2.plot(range(2, kmax + 1), linkage_distances, marker='o')
    ax2.axvline(x=k, c='red', lw=2, linestyle='dashed')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Linkage Distance')
    ax2.set_xticks(np.arange(2,kmax+1,1))
    ax2.tick_params(bottom=True, left=True)
    ax2.spines[:].set_linewidth(1)
    ax2.spines[:].set_color('k')
    plt.show()

def pca_var_contr(df, n_components, var_list, explained_ratio=True, stacked=False):
    model = PCA(n_components).fit(df)
    # Get loading scores
    corr = model.components_.T
    # Get explained ratio
    if explained_ratio:
        ratio = model.explained_variance_ratio_
    else:
        ratio = 1
    # Get PCs name
    pc_n = [f'PC{i+1}' for i in np.arange(np.shape(corr)[1])]
    # Compute feature contribution to each PC
    percentage_total = pd.DataFrame((np.abs(corr) / np.sum(np.abs(corr), axis=0))*100,columns=pc_n)*ratio
    # Get variables list
    percentage_total.index = var_list
    # Plot
    percentage_total.plot.barh(stacked=stacked)
    ax=plt.gca()
    ax.spines[:].set_linewidth(1)
    ax.spines[:].set_color('k')
    plt.tick_params(bottom=True, left=True)
    plt.xlabel('Explained variation (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Variable contribution to explained variation')
    plt.show()
    return(percentage_total)

def hca_silhouette_score(df,linkage_matrix,k,color):
    fig, axs = plt.subplots()
    # Perform hierarchical clustering and obtain labels
    cluster_labels = fcluster(linkage_matrix, t=k, criterion='maxclust')
    # Compute silhouette scores
    silhouette_avg = silhouette_score(df, cluster_labels).round(4)
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10 # gap between clusters
    for i in range(1, k + 1):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        missclass = np.shape(ith_cluster_silhouette_values[ith_cluster_silhouette_values < 0])[0]/np.shape(ith_cluster_silhouette_values)[0]
        y_upper = y_lower + size_cluster_i
        axs.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color[i-1], edgecolor=color[i-1], alpha=1)
        axs.text(-0.2, y_lower + 0.5 * size_cluster_i, f'(-{np.round(missclass,2)}%) {i}')
        y_lower = y_upper + 10
    axs.set_title(f"Avg silhouette score for {k} clusters is {silhouette_avg}")
    axs.set_ylabel("Cluster label")
    axs.axvline(x=silhouette_avg, color="red", linestyle="--")
    axs.tick_params(bottom=True)
    axs.set_yticks([])
    axs.spines[:].set_linewidth(1)
    axs.spines[:].set_color('k')
    plt.tight_layout()
    plt.show()
    return(sample_silhouette_values)
