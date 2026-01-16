"""
ML Model Marketplace Web Application
Platform for sharing, discovering, and deploying ML models
"""

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/ml_marketplace'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

logger = logging.getLogger(__name__)

# Database Models
class MLModel(db.Model):
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    version = db.Column(db.String(50), nullable=False)
    author = db.Column(db.String(255))
    category = db.Column(db.String(100))
    task_type = db.Column(db.String(100))  # classification, segmentation, detection
    framework = db.Column(db.String(50))  # pytorch, tensorflow, onnx
    model_path = db.Column(db.String(500))
    performance_metrics = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    downloads = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(50), default='pending')  # pending, approved, rejected

class ModelDeployment(db.Model):
    __tablename__ = 'model_deployments'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'))
    deployment_name = db.Column(db.String(255))
    endpoint_url = db.Column(db.String(500))
    status = db.Column(db.String(50))  # active, inactive, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    request_count = db.Column(db.Integer, default=0)
    avg_latency_ms = db.Column(db.Float)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    category = request.args.get('category')
    task_type = request.args.get('task_type')
    
    query = MLModel.query.filter_by(status='approved')
    
    if category:
        query = query.filter_by(category=category)
    if task_type:
        query = query.filter_by(task_type=task_type)
    
    models = query.order_by(MLModel.rating.desc()).all()
    
    return jsonify([{
        'id': m.id,
        'name': m.name,
        'description': m.description,
        'version': m.version,
        'author': m.author,
        'category': m.category,
        'rating': m.rating,
        'downloads': m.downloads
    } for m in models])

@app.route('/api/models/<int:model_id>', methods=['GET'])
def get_model(model_id):
    model = MLModel.query.get_or_404(model_id)
    
    return jsonify({
        'id': model.id,
        'name': model.name,
        'description': model.description,
        'version': model.version,
        'author': model.author,
        'category': model.category,
        'task_type': model.task_type,
        'framework': model.framework,
        'performance_metrics': model.performance_metrics,
        'rating': model.rating,
        'downloads': model.downloads,
        'created_at': model.created_at.isoformat()
    })

@app.route('/api/models/<int:model_id>/deploy', methods=['POST'])
def deploy_model(model_id):
    model = MLModel.query.get_or_404(model_id)
    data = request.json
    
    deployment = ModelDeployment(
        model_id=model_id,
        deployment_name=data.get('deployment_name'),
        status='active'
    )
    
    db.session.add(deployment)
    db.session.commit()
    
    return jsonify({
        'deployment_id': deployment.id,
        'status': 'success',
        'endpoint_url': f'/api/predict/{deployment.id}'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
