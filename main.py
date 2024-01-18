import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load deliveries dataset
deliveries_df = pd.read_csv('deliveries.csv')

# Load matches dataset
matches_df = pd.read_csv('matches.csv')

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
    df = df.dropna(axis='columns')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of numeric columns ({df.shape[1]}) is less than 2')
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

# Extracting relevant columns for analysis
season_deliveries_df = matches_df[['id', 'season', 'winner']]
complete_deliveries_df = deliveries_df.merge(season_deliveries_df, how='inner', left_on='match_id', right_on='id')
matches_df = matches_df.drop(columns=["umpire3"], axis=1)

# Number of Matches played in each IPL season
plt.figure(figsize=(18, 10))
sns.countplot(x='season', data=matches_df, palette="winter")
plt.title("Number of Matches played in each IPL season", fontsize=20)
plt.xlabel("season", fontsize=15)
plt.ylabel('Matches', fontsize=15)
plt.show()

# Numbers of matches won by team
plt.figure(figsize=(18, 10))
sns.countplot(x='winner', data=matches_df, palette='cool')
plt.title("Numbers of matches won by team", fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Teams", fontsize=15)
plt.ylabel("No of wins", fontsize=15)
plt.show()

# Match Result Pie Chart
matches_df['win_by'] = np.where(matches_df['win_by_runs'] > 0, 'Bat first', 'Bowl first')
Win = matches_df.win_by.value_counts()
labels = np.array(Win.index)
sizes = Win.values
colors = ['#FFBF00', '#FA8072']
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Match Result', fontsize=20)
plt.gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tick_params(labelsize=10)
plt.show()

# Numbers of matches won by batting and bowling first
plt.figure(figsize=(18, 10))
sns.countplot(x='season', hue='win_by', data=matches_df, palette='hsv')
plt.title("Numbers of matches won by batting and bowling first", fontsize=20)
plt.xlabel("Season", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

# Toss Decision Pie Chart
Toss = matches_df.toss_decision.value_counts()
labels = np.array(Toss.index)
sizes = Toss.values
colors = ['#FFBF00', '#FA8072']
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Toss Decision', fontsize=20)
plt.gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tick_params(labelsize=10)
plt.show()

# Numbers of matches won by Toss result
plt.figure(figsize=(18, 10))
sns.countplot(x='season', hue='toss_decision', data=matches_df, palette='afmhot')
plt.title("Numbers of matches won by Toss result", fontsize=20)
plt.xlabel("Season", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

# Winner season wise
final_matches = matches_df.drop_duplicates(subset=['season'], keep='last')
final_matches[['season', 'winner']].reset_index(drop=True).sort_values('season')

# Winning percentage in final Pie Chart
match = final_matches.win_by.value_counts()
labels = np.array(Toss.index)
sizes = match.values
colors = ['gold', 'purple']
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Winning percentage in final', fontsize=20)
plt.gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tick_params(labelsize=10)
plt.show()

# Toss Result Pie Chart
Toss = final_matches.toss_decision.value_counts()
labels = np.array(Toss.index)
sizes = Toss.values
colors = ['#FFBF00', '#FA8072']
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Toss Result', fontsize=20)
plt.gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tick_params(labelsize=10)
plt.show()

# Top player of the match Winners
plt.figure(figsize=(18, 10))
top_players = matches_df.player_of_match.value_counts()[:10]
fig, ax = plt.subplots()
ax.set_ylim([0, 20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
top_players.plot.bar()
sns.barplot(x=top_players.index, y=top_players, orient='v', palette="hsv")
plt.show()

# IPL Finals venues and winners along with the number of wins
final_matches.groupby(['city', 'winner']).size()

# Number of seasons won by teams
final_matches["winner"].value_counts()

# Toss winner, toss decision, winner in final matches
final_matches[['toss_winner', 'toss_decision', 'winner']].reset_index(drop=True)

# Man of the match
final_matches[['winner', 'player_of_match']].reset_index(drop=True)

# Number of fours hit by team
four_deliveries_df = complete_deliveries_df[complete_deliveries_df['batsman_runs'] == 4]
fours_by_team = four_deliveries_df.groupby('batting_team')['batsman_runs'].agg([('runs by fours', 'sum'), ('fours', 'count')])

# Graph on four hits by players
batsman_four = four_deliveries_df.groupby('batsman')['batsman_runs'].agg([('four', 'count')]).reset_index().sort_values('four', ascending=0)
ax = batsman_four.iloc[:10, :].plot('batsman', 'four', kind='bar', color='black')
plt.title("Numbers of fours hit by players", fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Player name", fontsize=15)
plt.ylabel("No of fours", fontsize=15)
plt.show()

# Graph on number of fours hit in each season
ax = four_deliveries_df.groupby('season')['batsman_runs'].agg([('four', 'count')]).reset_index().plot('season', 'four', kind='bar', color='red')
plt.title("Numbers of fours hit in each season", fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Season", fontsize=15)
plt.ylabel("No of fours", fontsize=15)
plt.show()

# Number of sixes hit by team
six_deliveries_df = complete_deliveries_df[complete_deliveries_df['batsman_runs'] == 6]
sixes_by_team = six_deliveries_df.groupby('batting_team')['batsman_runs'].agg([('runs by six', 'sum'), ('sixes', 'count')])

# Graph of six hits by players
batsman_six = six_deliveries_df.groupby('batsman')['batsman_runs'].agg([('six', 'count')]).reset_index().sort_values('six', ascending=0)
ax = batsman_six.iloc[:10, :].plot('batsman', 'six', kind='bar', color='green')
plt.title("Numbers of six hit by players", fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Player name", fontsize=15)
plt.ylabel("No of six", fontsize=15)
plt.show()

# Graph on number of six hits in each season
ax = six_deliveries_df.groupby('season')['batsman_runs'].agg([('six', 'count')]).reset_index().plot('season', 'six', kind='bar', color='blue')
plt.title("Numbers of six hit in each season", fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Season", fontsize=15)
plt.ylabel("No of six", fontsize=15)
plt.show()

# Top 10 leading run scorers in IPL
batsman_score = deliveries_df.groupby('batsman')['batsman_runs'].agg(['sum']).reset_index().sort_values('sum', ascending=False).reset_index(drop=True)
batsman_score = batsman_score.rename(columns={'sum': 'batsman_runs'})
print("*** Top 10 Leading Run Scorers in IPL ***")
batsman_score.iloc[:10, :]

# Number of matches played by batsman
No_Matches_player = deliveries_df[["match_id", "player_dismissed"]]
No_Matches_player = No_Matches_player.groupby("player_dismissed")["match_id"].count().reset_index().sort_values(by="match_id", ascending=False).reset_index(drop=True)
No_Matches_player.columns = ["batsman", "No_of Matches"]
No_Matches_player.head(5)

# Dismissals in IPL
plt.figure(figsize=(18, 10))
ax = sns.countplot(deliveries_df.dismissal_kind)
plt.title("Dismissals in IPL", fontsize=20)
plt.xlabel("Dismissals kind", fontsize=15)
plt.ylabel("count", fontsize=15)
plt.xticks(rotation=90)
plt.show()

# IPL most wicket-taking bowlers
wicket_deliveries_df = deliveries_df.dropna(subset=['dismissal_kind'])
wicket_deliveries_df = wicket_deliveries_df[~wicket_deliveries_df['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
most_wickets = wicket_deliveries_df.groupby('bowler')['dismissal_kind'].agg(['count']).reset_index().sort_values('count', ascending=False).reset_index(drop=True).iloc[:10, :]

# Identify most successful teams
most_successful_teams = matches_df['winner'].value_counts().head(5)
print("Most Successful Teams:")
print(most_successful_teams)

# Identify most successful players
most_successful_players = deliveries_df.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(5)
print("\nMost Successful Players:")
print(most_successful_players)

# Visualize the most successful teams
plt.figure(figsize=(12, 6))
most_successful_teams.plot(kind='bar', color='cyan')
plt.title('Most Successful Teams')
plt.xlabel('Teams')
plt.ylabel('Total Wins')
plt.show()

# Visualize the most successful players
plt.figure(figsize=(12, 6))
most_successful_players.plot(kind='bar', color='magenta')
plt.title('Most Successful Players')
plt.xlabel('Players')
plt.ylabel('Total Runs')
plt.show()


