import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load deliveries dataset
deliveries_df = pd.read_csv('/content/deliveries.csv')

# Load matches dataset
matches_df = pd.read_csv('/content/matches.csv')

# Explore the structure of the datasets
print("Deliveries Dataset:")
print(deliveries_df.info())
print("\nMatches Dataset:")
print(matches_df.info())

# Distribution graphs for deliveries dataset
def plot_per_column_distribution(df, n_graph_shown, n_graph_per_row):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    n_row, n_col = df.shape
    column_names = list(df)
    n_graph_row = (n_col + n_graph_per_row - 1) // n_graph_per_row
    plt.figure(num=None, figsize=(6 * n_graph_per_row, 8 * n_graph_row), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(n_col, n_graph_shown)):
        plt.subplot(n_graph_row, n_graph_per_row, i + 1)
        column_df = df.iloc[:, i]
        if not pd.api.types.is_numeric_dtype(column_df.dtype):
            value_counts = column_df.value_counts()
            value_counts.plot.bar()
        else:
            column_df.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{column_names[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

# Correlation matrix for matches dataset
def plot_correlation_matrix(df, graph_width):
    df = df.dropna(axis='columns')  # Use axis parameter for future versions
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graph_width, graph_width), dpi=80, facecolor='w', edgecolor='k')
    corr_mat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corr_mat)
    plt.title(f'Correlation Matrix for Matches Dataset', fontsize=15)
    plt.show()

# Scatter and density plots for deliveries dataset
def plot_scatter_matrix(df, plot_size, text_size):
    df = df.select_dtypes(include=[float, int])  # Use specific types instead of deprecated np.number
    df = df.dropna(axis='columns')  # Use axis parameter for future versions
    df = df[[col for col in df if df[col].nunique() > 1]]
    column_names = list(df)
    if len(column_names) > 10:
        column_names = column_names[:10]
    df = df[column_names]
    sns.set(style="ticks")
    sns.pairplot(df, height=plot_size)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

# Perform EDA on deliveries dataset
plot_per_column_distribution(deliveries_df, 10, 5)
plot_correlation_matrix(deliveries_df, 8)
plot_scatter_matrix(deliveries_df, 20, 10)

# Perform EDA on matches dataset
plot_per_column_distribution(matches_df, 10, 5)
plot_correlation_matrix(matches_df, 8)

# Factors Contributing to Wins/Losses
team_wins = matches_df['winner'].value_counts()
# Toss impact on match outcomes
plt.figure(figsize=(12, 6))
sns.countplot(x='season', hue='toss_winner', data=matches_df, hue_order=team_wins.index)
plt.title('Impact of Toss on Match Outcomes Across Seasons')
plt.xlabel('Seasons')
plt.ylabel('Total Matches')
plt.legend(title='Toss Winner', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Correlation between team totals and match results
team_totals = deliveries_df.groupby('match_id')['total_runs'].sum().reset_index()
team_totals = team_totals.merge(matches_df[['id', 'result']], left_on='match_id', right_on='id')

plt.figure(figsize=(12, 6))
sns.boxplot(x='result', y='total_runs', data=team_totals)
plt.title('Correlation Between Team Totals and Match Results')
plt.xlabel('Match Result')
plt.ylabel('Total Runs')
plt.show()

# Player Performance Analysis
top_run_scorers = deliveries_df.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)
top_wicket_takers = deliveries_df[deliveries_df['dismissal_kind'].notna()].groupby('bowler')['dismissal_kind'].count().sort_values(ascending=False).head(10)

# Visualize top run-scorers
top_run_scorers.plot(kind='bar', figsize=(12, 6), color='orange')
plt.title('Top Run Scorers in IPL')
plt.xlabel('Players')
plt.ylabel('Total Runs')
plt.show()

# Visualize top wicket-takers
top_wicket_takers.plot(kind='bar', figsize=(12, 6), color='purple')
plt.title('Top Wicket Takers in IPL')
plt.xlabel('Players')
plt.ylabel('Total Wickets')
plt.show()

# Distribution of runs scored
plt.figure(figsize=(12, 6))
sns.histplot(deliveries_df.groupby('batsman')['batsman_runs'].sum(), bins=50, kde=True, color='blue')
plt.title('Distribution of Runs Scored by Players')
plt.xlabel('Total Runs')
plt.ylabel('Frequency')
plt.show()

# Distribution of wickets taken
plt.figure(figsize=(12, 6))
sns.histplot(deliveries_df[deliveries_df['dismissal_kind'].notna()].groupby('bowler')['dismissal_kind'].count(), bins=50, kde=True, color='red')
plt.title('Distribution of Wickets Taken by Bowlers')
plt.xlabel('Total Wickets')
plt.ylabel('Frequency')
plt.show()

# Batting and Bowling averages of players
batting_average = deliveries_df.groupby('batsman')['batsman_runs'].mean()

# Check the columns available in deliveries_df to confirm the correct names
print(deliveries_df.columns)

# Assuming 'dismissal_kind' is the column indicating type of dismissal
# and 'player_dismissed' is the column indicating the player dismissed
bowling_average = deliveries_df[deliveries_df['dismissal_kind'].notna()].groupby('player_dismissed')['dismissal_kind'].count() / deliveries_df[deliveries_df['dismissal_kind'].notna()].groupby('player_dismissed')['dismissal_kind'].count().sum()

# Plotting
plt.figure(figsize=(12, 6))
sns.histplot(batting_average, bins=50, kde=True, color='green', label='Batting Average')
sns.histplot(bowling_average, bins=50, kde=True, color='orange', label='Bowling Average')
plt.title('Batting and Bowling Averages of Players')
plt.xlabel('Average')
plt.ylabel('Frequency')
plt.legend()
plt.show()
