from flask import Flask, render_template, request, jsonify
import requests


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    if request.method == 'GET':
        # Get user_id entered by user in the dashboard
        user_id = request.args.get('user_id')
        print('user_id: ', user_id)
        url = "https://based-content.azurewebsites.net/api/based-content-function"
        recommandations = requests.get(url, params={'user_id': user_id}).json()['recommandation_articles']
    return render_template('result.html', user_id=user_id, recommandations=recommandations)
