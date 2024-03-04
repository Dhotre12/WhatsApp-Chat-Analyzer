import unittest
from unittest.mock import patch
from app import app
from preprocessor import preprocess
from helper import fetch_stats, generate_wordcloud, generate_emoji_pie_chart, generate_activity_heatmap, generate_busiest_day_bar_graph, generate_busiest_month_bar_graph, generate_user_activity_bar_graph, generate_common_words_bar_graph

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    # Add more test cases for other routes...

    def test_preprocess(self):
        # Write test cases for preprocess function
        pass

    def test_fetch_stats(self):
        # Write test cases for fetch_stats function
        pass

    # Add more test cases for other functions...

if __name__ == '__main__':
    unittest.main()
