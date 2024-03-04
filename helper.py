import pandas as pd
from wordcloud import WordCloud
import re
from collections import Counter
import matplotlib.pyplot as plt
import emoji
import atexit
import seaborn as sns
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Register cleanup function using atexit module
atexit.register(plt.close)

def fetch_stats(df):
    # Placeholder logic
    num_messages = df.shape[0]
    words = df['message'].apply(lambda x: len(str(x).split())).sum()
    num_media_messages = df[df['message'].str.contains('<Media omitted>')].shape[0]
    num_links = df[df['message'].str.contains('http')].shape[0]

    total_users = df['user'].nunique()-1
    total_messages_per_day = num_messages / ((pd.to_datetime('today') - df['date'].min()).days + 1)

    # Calculate most talkative user
    most_talkative_user = df['user'].value_counts().idxmax()

    return {
        'num_messages': num_messages,
        'words': words,
        'num_media_messages': num_media_messages,
        'num_links': num_links,
        'total_users': total_users,
        'total_messages_per_day': "{:.2f}".format(total_messages_per_day),
        'most_talkative_user': most_talkative_user
    }


def generate_wordcloud(df):
    # Concatenate all messages into a single string
    all_messages = ' '.join(df['message'].values.tolist())

    # Remove unwanted words like "media omitted"
    all_messages = re.sub(r'\bmedia omitted\b', '', all_messages, flags=re.IGNORECASE)
    all_messages = re.sub(r'\bmessage deleted\b', '', all_messages, flags=re.IGNORECASE)
    all_messages = re.sub(r'\bmessage\b', '', all_messages, flags=re.IGNORECASE)
    all_messages = re.sub(r'\bdeleted\b', '', all_messages, flags=re.IGNORECASE)
   
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    # Save the word cloud image to a file
    wordcloud_image_path = 'static/wordcloud.png'
    plt.savefig(wordcloud_image_path)

    # Close the plot to free up memory
    plt.close()
    
    return wordcloud_image_path

def generate_emoji_pie_chart(df, top_emoji_count=10, font_size=12):
    """
    Generates a pie chart showing the top `top_emoji_count` most used emojis.

    Args:
        df (pandas.DataFrame): DataFrame containing a 'message' column.
        top_emoji_count (int, optional): Number of top emojis to show. Defaults to 6.
        font_size (int, optional): Font size for labels and percentages. Defaults to 12.

    Returns:
        tuple: Path to the saved pie chart image and list of emojis.
    """

    try:
        # Ensure `get_emoji_regexp` is available (assuming emoji >= 1.4.0)
        emojis = emoji.get_emoji_regexp().findall(' '.join(df['message']))
        emoji_counts = Counter(emojis)

        # Limit to top emojis
        top_emojis = emoji_counts.most_common(top_emoji_count)

        # Extract labels and values
        labels = [k for k, _ in top_emojis]
        values = [v for _, v in top_emojis]

        # Create and save the pie chart
        fig = px.pie(names=labels, values=values, title='Mostly Used Emoji', labels={'names': 'Emoji'})
        fig.update_traces(textposition='inside', textinfo='percent+label')

        # Save the pie chart as an image
        emoji_chart_image_path = 'static/emoji_chart.png'
        fig.write_image(emoji_chart_image_path)

        # Prepare emoji list
        emoji_list = [(k + ' ' + emoji.demojize(k), v) for k, v in emoji_counts.items()]

        return emoji_chart_image_path, emoji_list

    except (KeyError, AttributeError) as e:
        print(f"Error generating emoji chart: {e}")
        return None, None  # Indicate error, handle accordingly





def generate_busiest_day_bar_graph(df):
    try:
        # Convert the date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Extract day of the week from the date
        df['day_of_week'] = df['date'].dt.day_name()

        # Count the number of messages for each day of the week
        day_counts = df['day_of_week'].value_counts().sort_index()

        # Reindex to include all days of the week in correct order
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = day_counts.reindex(all_days, fill_value=0)

        # Plot the bar graph
        plt.figure(figsize=(8, 6))
        day_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot as an image
        busiest_day_image_path = 'static/busiest_day_bar_graph.png'
        plt.savefig(busiest_day_image_path)

        # Close the plot to free up memory
        plt.close()

        return busiest_day_image_path

    except Exception as e:
        print(f"Error generating busiest day bar graph: {e}")
        return None


def generate_busiest_month_bar_graph(df):
    try:
        # Convert the date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Extract month from the date
        df['month'] = df['date'].dt.month_name()

        # Count the number of messages for each month
        month_counts = df['month'].value_counts().sort_index()

        # Reindex to include all months in correct order
        all_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        month_counts = month_counts.reindex(all_months, fill_value=0)

        # Plot the bar graph
        plt.figure(figsize=(10, 6))
        month_counts.plot(kind='bar', color='lightgreen')
        plt.xlabel('Month')
        plt.ylabel('Number of Messages')
        # plt.title('Monthly Activity')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot as an image
        busiest_month_image_path = 'static/busiest_month_bar_graph.png'
        plt.savefig(busiest_month_image_path)

        # Close the plot to free up memory
        plt.close()

        return busiest_month_image_path

    except Exception as e:
        print(f"Error generating busiest month bar graph: {e}")
        return None


def generate_activity_heatmap(df):
    # Pivot the DataFrame to get the count of messages per hour for each day
    activity_data = df.pivot_table(index='day_of_week', columns='hour', aggfunc='size', fill_value=0)

    # Reorder the days of the week for proper visualization
    activity_data = activity_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(activity_data, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
    # plt.title('Message Activity Heatmap')
    plt.xlabel('Hour')
    plt.ylabel('Day of the Week')
    plt.tight_layout()
    plt.savefig('static/activity_heatmap.png')  # Save the heatmap image
    plt.close()

    return 'static/activity_heatmap.png'

def generate_user_activity_bar_graph(df):
    try:
        # Filter out 'group_notification' users
        filtered_df = df[df['user'] != 'group_notification']

        # Calculate total messages per user
        user_messages = filtered_df['user'].value_counts()

        # Calculate the number of weeks covered in the chat data
        num_weeks = (filtered_df['date'].max() - filtered_df['date'].min()).days // 7 + 1

        # Calculate average messages per week for each user
        user_avg_messages_per_week = user_messages / num_weeks

        # Select top 10 users with highest average messages per week
        top_users = user_avg_messages_per_week.nlargest(10)

        # Plot the bar graph
        plt.figure(figsize=(10, 6))
        top_users.plot(kind='bar', color='salmon')
        plt.xlabel('User')
        plt.ylabel('Average Messages per Week')
        plt.title('Top 10 Most Active Users')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot as an image
        user_activity_image_path = 'static/user_activity_bar_graph.png'
        plt.savefig(user_activity_image_path)
        plt.close()  # Close the plot

        return user_activity_image_path

    except Exception as e:
        print(f"Error generating user activity bar graph: {e}")
        return None


def generate_common_words_bar_graph(df):
    # Preprocess messages
    messages = df['message'].str.lower()

    # Remove punctuation and symbols
    messages = messages.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Tokenize words
    words = ' '.join(messages).split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Define additional words or phrases to remove
    words_to_remove = ['media', 'omitted', 'message', 'deleted', '<>']  # Add more words as needed

    # Remove specific words or phrases
    words = [word for word in words if word not in words_to_remove]

    # Calculate word frequency
    word_freq = pd.Series(words).value_counts().reset_index()
    word_freq.columns = ['word', 'frequency']

    # Take top 10 common words
    top_10_words = word_freq.head(10)

    # Plot bar graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x='frequency', y='word', data=top_10_words, palette='viridis')
    plt.title('Top 10 Common Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()

    # Save bar graph image
    common_words_bar_graph_path = 'static/common_words_bar_graph.png'
    plt.savefig(common_words_bar_graph_path)
    plt.close()  # Close the plot to free memory

    return common_words_bar_graph_path