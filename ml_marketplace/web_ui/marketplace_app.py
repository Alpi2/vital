#!/usr/bin/env python3
"""
ML Model Marketplace Web Application
Provides UI for browsing, deploying, and managing ML models
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ml-marketplace-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://vital_user:vital_password@localhost/ml_marketplace'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)

# Database Models
class MLModel(db.Model):
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    model_type = db.Column(db.String(100))
    framework = db.Column(db.String(50))
    task = db.Column(db.String(100))
    accuracy = db.Column(db.Float)
    latency_ms = db.Column(db.Float)
    model_size_mb = db.Column(db.Float)
    docker_image = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    downloads = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(50), default='active')
    
class ModelDeployment(db.Model):
    __tablename__ = 'model_deployments'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'))
    user_id = db.Column(db.Integer)
    environment = db.Column(db.String(50))
    endpoint_url = db.Column(db.String(500))
    status = db.Column(db.String(50))
    deployed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class ABTest(db.Model):
    __tablename__ = 'ab_tests'
    
    id = db.Column(db.Integer, primary_key=True)
    test_name = db.Column(db.String(200))
    model_a_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'))
    model_b_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'))
    traffic_split = db.Column(db.Float, default=0.5)
    status = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class ModelMetrics(db.Model):
    __tablename__ = 'model_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'))
    metric_name = db.Column(db.String(100))
    metric_value = db.Column(db.Float)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)

# Routes
@app.route('/')
def index():
    """Homepage - Browse models"""
    models = MLModel.query.filter_by(status='active').order_by(MLModel.downloads.desc()).limit(12).all()
    return render_template('marketplace/index.html', models=models)

@app.route('/models')
def browse_models():
    """Browse all models with filters"""
    task = request.args.get('task')
    framework = request.args.get('framework')
    sort_by = request.args.get('sort', 'downloads')
    
    query = MLModel.query.filter_by(status='active')
    
    if task:
        query = query.filter_by(task=task)
    if framework:
        query = query.filter_by(framework=framework)
    
    if sort_by == 'downloads':
        query = query.order_by(MLModel.downloads.desc())
    elif sort_by == 'rating':
        query = query.order_by(MLModel.rating.desc())
    elif sort_by == 'recent':
        query = query.order_by(MLModel.created_at.desc())
    
    models = query.all()
    return render_template('marketplace/browse.html', models=models)

@app.route('/models/<int:model_id>')
def model_detail(model_id):
    """Model detail page"""
    model = MLModel.query.get_or_404(model_id)
    metrics = ModelMetrics.query.filter_by(model_id=model_id).all()
    deployments = ModelDeployment.query.filter_by(model_id=model_id).all()
    
    return render_template('marketplace/model_detail.html', 
                         model=model, 
                         metrics=metrics,
                         deployments=deployments)

@app.route('/api/models', methods=['GET'])
def api_list_models():
    """API endpoint to list models"""
    models = MLModel.query.filter_by(status='active').all()
    return jsonify([{
        'id': m.id,
        'name': m.name,
        'version': m.version,
        'description': m.description,
        'task': m.task,
        'framework': m.framework,
        'accuracy': m.accuracy,
        'latency_ms': m.latency_ms,
        'downloads': m.downloads,
        'rating': m.rating
    } for m in models])

@app.route('/api/models/<int:model_id>', methods=['GET'])
def api_get_model(model_id):
    """API endpoint to get model details"""
    model = MLModel.query.get_or_404(model_id)
    return jsonify({
        'id': model.id,
        'name': model.name,
        'version': model.version,
        'description': model.description,
        'model_type': model.model_type,
        'framework': model.framework,
        'task': model.task,
        'accuracy': model.accuracy,
        'latency_ms': model.latency_ms,
        'model_size_mb': model.model_size_mb,
        'docker_image': model.docker_image,
        'downloads': model.downloads,
        'rating': model.rating,
        'created_at': model.created_at.isoformat(),
        'updated_at': model.updated_at.isoformat()
    })

@app.route('/api/models/<int:model_id>/deploy', methods=['POST'])
@login_required
def api_deploy_model(model_id):
    """Deploy model to environment"""
    model = MLModel.query.get_or_404(model_id)
    data = request.get_json()
    
    environment = data.get('environment', 'staging')
    
    # Create deployment record
    deployment = ModelDeployment(
        model_id=model_id,
        user_id=current_user.id,
        environment=environment,
        endpoint_url=f"https://api.vital.ai/models/{model_id}/predict",
        status='deploying'
    )
    
    db.session.add(deployment)
    db.session.commit()
    
    # Trigger deployment process (async)
    # deploy_model_async.delay(deployment.id)
    
    return jsonify({
        'deployment_id': deployment.id,
        'status': 'deploying',
        'endpoint_url': deployment.endpoint_url
    }), 202

@app.route('/api/ab-tests', methods=['POST'])
@login_required
def api_create_ab_test():
    """Create A/B test"""
    data = request.get_json()
    
    ab_test = ABTest(
        test_name=data['test_name'],
        model_a_id=data['model_a_id'],
        model_b_id=data['model_b_id'],
        traffic_split=data.get('traffic_split', 0.5),
        status='active'
    )
    
    db.session.add(ab_test)
    db.session.commit()
    
    return jsonify({
        'test_id': ab_test.id,
        'status': 'active',
        'message': 'A/B test created successfully'
    }), 201

@app.route('/api/ab-tests/<int:test_id>/results', methods=['GET'])
def api_get_ab_test_results(test_id):
    """Get A/B test results"""
    ab_test = ABTest.query.get_or_404(test_id)
    
    # Fetch metrics for both models
    model_a_metrics = ModelMetrics.query.filter_by(model_id=ab_test.model_a_id).all()
    model_b_metrics = ModelMetrics.query.filter_by(model_id=ab_test.model_b_id).all()
    
    return jsonify({
        'test_id': ab_test.id,
        'test_name': ab_test.test_name,
        'model_a': {
            'id': ab_test.model_a_id,
            'metrics': [{'name': m.metric_name, 'value': m.metric_value} for m in model_a_metrics]
        },
        'model_b': {
            'id': ab_test.model_b_id,
            'metrics': [{'name': m.metric_name, 'value': m.metric_value} for m in model_b_metrics]
        },
        'traffic_split': ab_test.traffic_split,
        'status': ab_test.status
    })

@app.route('/api/models/<int:model_id>/metrics', methods=['POST'])
def api_record_metric(model_id):
    """Record model metric"""
    data = request.get_json()
    
    metric = ModelMetrics(
        model_id=model_id,
        metric_name=data['metric_name'],
        metric_value=data['metric_value']
    )
    
    db.session.add(metric)
    db.session.commit()
    
    return jsonify({'message': 'Metric recorded'}), 201

@app.route('/deployments')
@login_required
def my_deployments():
    """User's model deployments"""
    deployments = ModelDeployment.query.filter_by(user_id=current_user.id).all()
    return render_template('marketplace/deployments.html', deployments=deployments)

@app.route('/ab-tests')
@login_required
def ab_tests():
    """A/B testing dashboard"""
    tests = ABTest.query.all()
    return render_template('marketplace/ab_tests.html', tests=tests)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=False)
