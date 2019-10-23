import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def get_most_correlated_variables(corr, num_pairs=10):
    correlation_melted = pd.melt(corr.reset_index().rename(columns={"index": "var_1"}), id_vars=("var_1"),var_name='var_2')
    correlation_melted = correlation_melted[correlation_melted.var_1!=correlation_melted.var_2]
    correlation_melted['var_couple'] = correlation_melted[['var_1','var_2']].apply(lambda x:tuple(sorted([x[0],x[1]])), axis=1)
    correlation_melted = correlation_melted.drop_duplicates(subset='var_couple').drop(['var_couple'],axis=1)
    correlation_melted['abs_value'] = correlation_melted['value'].abs().round(3)
    return correlation_melted.sort_values(by='abs_value').tail(num_pairs).drop('abs_value', axis=1).reset_index(drop=True)

def plot_correlation_matrix(X, features2):
    corr = X[features2].corr()
     # return the most correlated variables
    most_correlated_variables = get_most_correlated_variables(corr, num_pairs=10)
    max_correlation = 1.25*most_correlated_variables['value'].abs().max()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 25))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax.set_yticklabels(features2, fontsize=18)
    ax.set_xticklabels(features2, rotation='vertical', fontsize=18)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, #vmax=max_correlation
                square=True, linewidths=.5, cbar_kws={"shrink": .8})

    return most_correlated_variables, corr

def color_negative_red(val):
    color = 'red' if np.abs(val) > 0.5 else 'black'
    return 'color: {}' .format(color)

def feature_importance(df, features, label, criterion='gini', plot_importance=False):
    # Set the features and the label for the random forest
    df_features = df[features].copy()
    target = df[label]
    # Create a random forest classifier
    forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1, criterion=criterion)
    # Train the classifier
    print('df_features dtypes', df_features.dtypes)
    forest = forest.fit(df_features, target)

    # Get the feature's importance
    importances_basic = forest.feature_importances_
    indices = np.argsort(importances_basic)[::-1]

    # Print the feature ranking
    importances_dict = {}
    if plot_importance:
        print("Feature ranking:")
    for i, feature in enumerate(df_features.columns[indices]):
        importances_dict[feature] = importances_basic[indices][i] * 100
        if plot_importance:
            print("{}. {} ({:.2f}%)".format(i + 1, feature, importances_basic[indices][i] * 100))

    if plot_importance:
        # Plot the feature importances
        plt.figure(figsize=(10, 7))
        plt.title("Feature importances")
        plt.bar(range(df_features.shape[1]), importances_basic[indices],
                color="red", align="center")

        plt.ylabel('Importance')
        plt.xlabel('Features')
        plt.xticks(range(df_features.shape[1]), df_features.columns[indices], rotation='vertical')
        plt.xlim([-1, df_features.shape[1]])
        plt.show()
    return importances_dict


def kde_plots(df, features, label, text_labels):
    # Generates kde plots for the df features, regading the label
    sns.set_style("darkgrid")
    plt.figure(figsize=(14, 4 * len(features)))
    if label == 'Support':
        a = -1
        b = 0
        c = 1
    else:
        a = 0
        b = 1
        c = 2
    for i, col in enumerate(features):
        plt.subplot2grid((len(features), 1), (i, 0))
        # plt.subplot(len(features),1,i+1, rowspan=1)
        sns.kdeplot(df.loc[df[label] == a, col], label=text_labels[0], color='r')
        sns.kdeplot(df.loc[df[label] == b, col], label=text_labels[1], color='black')
        sns.kdeplot(df.loc[df[label] == c, col], label=text_labels[2], color='g')
        plt.ylabel('Density estimate')
        plt.xlabel('Feature values')
        plt.legend(fontsize=22)
        plt.title(col, fontsize=22)
        plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=1.0)
    # plt.subsubplots_adjust(hspace = 1)
    plt.show()


def gen_df_hist(df, features, n_bins=100):
    # Generates histograms for the df features
    plt.figure(figsize=(20, 4 * len(features)))
    for i, col in enumerate(features):
        plt.subplot2grid((len(features), 1), (i, 0))
        # plt.subplot(len(features),1,i+1)
        plt.hist(df[col], density=True, rwidth=0.8, bins=n_bins)
        min_val = df[col].min()
        max_val = df[col].max()
        plt.xlim(min_val, max_val)
        # pd.DataFrame.hist(df, column=col, bins=100 ,figsize=(20,10))
        plt.title('{} Histogram'.format(col))  # , fontsize=20)
        plt.ylabel('Frequency')  # , fontsize=16)
        plt.xlabel(col)  # , fontsize=16)
    # plt.subsubplots_adjust(hspace = 1)
    plt.show()


def gen_df_density(df, features):
    # Plots density
    for i, col in enumerate(features):
        # plt.subplot2grid((len(features), 1), (i, 0))
        plt.figure(figsize=(20, 4))
        sns.distplot(df[col], hist=False, rug=True)
        plt.xlim(0)
        plt.xlabel(col, fontsize=16)
        plt.ylabel('Probability', fontsize=16)
        plt.title('{} distribution'.format(col), fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()