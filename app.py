# app.py

from flask import Flask, render_template, request, send_file
from preprocessor import preprocess
from helper import fetch_stats, generate_wordcloud, generate_emoji_pie_chart, generate_activity_heatmap, generate_busiest_day_bar_graph, generate_busiest_month_bar_graph, generate_user_activity_bar_graph, generate_common_words_bar_graph 
import pandas as pd
from collections import Counter


app = Flask(__name__)

df = None  # Global DataFrame variable

@app.route('/', methods=['GET', 'POST'])
def index():
    global df

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            try:
                bytes_data = file.read()
                df = preprocess(bytes_data)
                stats = fetch_stats(df)
                wordcloud_image = generate_wordcloud(df)
                emoji_chart_image_path, emoji_list = generate_emoji_pie_chart(df)
                busiest_day_bar_graph_path = generate_busiest_day_bar_graph(df)
                busiest_month_bar_graph_path = generate_busiest_month_bar_graph(df)
                activity_heatmap_image = generate_activity_heatmap(df)
                user_activity_bar_graph_path = generate_user_activity_bar_graph(df)  # New graph
                common_words_bar_graph_path = generate_common_words_bar_graph(df)  # New graph

                # Calculate the most talkative user and their message count
                user_message_counts = Counter(df['user'])
                most_talkative_user = user_message_counts.most_common(1)[0][0]
                most_talkative_message_count = user_message_counts[most_talkative_user]


                return render_template('result.html',
                                       emoji_chart_image_path=emoji_chart_image_path,
                                       emoji_list=emoji_list,
                                       wordcloud_image=wordcloud_image,
                                       stats=stats,
                                       busiest_day_bar_graph_path=busiest_day_bar_graph_path,
                                       busiest_month_bar_graph_path=busiest_month_bar_graph_path,
                                       activity_heatmap_image=activity_heatmap_image,
                                       user_activity_bar_graph_path=user_activity_bar_graph_path,  # Include new graph paths
                                       common_words_bar_graph_path=common_words_bar_graph_path,  # Include new graph paths
                                       most_talkative_user=most_talkative_user,
                                       most_talkative_message_count=most_talkative_message_count)


            except Exception as e:
                print("Error:", e)
                return render_template('index.html', error="An error occurred during processing")

    return render_template('index.html')

@app.route('/activity_percentage', methods=['POST'])
def activity_percentage():
    user = request.form['user']
    activity = request.form['activity']

    if df is None:
        return render_template('result.html', error="DataFrame is not loaded yet")

    activity_percentage = generate_activity_heatmap(df, user, activity)
    return render_template('result.html', user=user, activity=activity, activity_percentage=activity_percentage)

@app.route('/get_heatmap')
def get_heatmap():
    return send_file('static/activity_heatmap.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
