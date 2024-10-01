from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from app.routes.routes import register_routes

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')

    CORS(app)

    api = Api(app)

    register_routes(api)

    return app