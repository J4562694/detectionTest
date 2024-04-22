from flask import Blueprint

apiRouter = Blueprint('apiRouter', __name__)

from app.module.apiRoutes.hello import router as helloRouter
apiRouter.register_blueprint(helloRouter, url_prefix='/hello')