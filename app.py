"""

app.py

# todo do this...
# todo production...
# todo dotenv
# todo /
# todo clean up folders

"""

from itertools import chain
from flask import Flask, abort
from flask import request

from compute import predict_user_topN
from compute import predict_user_controversial_items
from compute import predict_user_hate_items
from compute import predict_user_hip_items
from compute import predict_user_no_clue_items

from models import Rating

app = Flask(__name__)


@app.route('/preferences', methods=['POST'])
def predict_preferences():
    req = request.json

    ratings = None

    try:
        ratings = req['ratings']
    except KeyError:
        abort(400)

    funcs = {
        'top_n': predict_user_topN,
        'controversial': predict_user_controversial_items,
        'hate': predict_user_hate_items,
        'hip': predict_user_hip_items,
        'no_clue': predict_user_no_clue_items
    }

    ratings = [Rating(**rating) for rating in ratings]

    predictions = list(chain( # flatten list of dicts
        *[f(ratings=ratings, user_id=0) for f in funcs.values()]))

    return dict(preferences=predictions)


if __name__ == '__main__':
    app.run()
