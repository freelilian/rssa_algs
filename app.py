from flask import Flask
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

    print("Testing")
    req = request.json
    user_id = req['user_id']
    ratings = req['ratings']

    # convert ratings from dict to a data class, tested using Postman
    ratings = [Rating(rating['item_id'], rating['rating']) for rating in ratings]

    preds = {
        'top_n': predict_user_topN,
        'hate': predict_user_hate_items,
        'hip': predict_user_hip_items,
        'no_clue': predict_user_no_clue_items,
        'controversial': predict_user_controversial_items
    }

    for key, func in preds.items():
        preds[key] = func(ratings=ratings, user_id=user_id)

    print(preds)
    return preds


if __name__ == '__main__':
    app.run()
