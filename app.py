import boto3
import json
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'fraud-detection-endpoint')
REGION = os.environ.get('AWS_REGION', 'eu-north-1')

# Initialize SageMaker Runtime Client
try:
    runtime = boto3.client('sagemaker-runtime', region_name=REGION)
    print(f"SageMaker Runtime initialized for region: {REGION}")
except Exception as e:
    print(f"Error initializing SageMaker client: {e}")
    runtime = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if runtime:
        return jsonify({
            'status': 'healthy',
            'service': 'fraud-detection-gateway',
            'endpoint': ENDPOINT_NAME
        }), 200
    else:
        return jsonify({'status': 'unhealthy', 'error': 'AWS client not initialized'}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Proxy endpoint to SageMaker Fraud Detection Model.
    Accepts JSON input (dict or list of dicts) and forwards it to SageMaker.
    """
    if not runtime:
        return jsonify({'error': 'SageMaker client not available'}), 503

    try:
        # Get JSON data from request
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'Invalid JSON input'}), 400

        # Invoke SageMaker Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(data)
        )

        # Read and decode the response
        payload = response['Body'].read().decode('utf-8')
        result = json.loads(payload)

        # Add risk assessment to the response
        if 'predictions' in result:
            predictions = result['predictions']
            risk_assessments = []

            for score in predictions:
                score = float(score)
                level = "LOW"
                if score > 0.5:
                    level = "HIGH"
                elif score > 0.2:
                    level = "MEDIUM"

                risk_assessments.append({
                    'fraud_probability': score,
                    'risk_level': level
                })

            result['risk_assessment'] = risk_assessments

        return jsonify(result)

    except boto3.exceptions.Boto3Error as e:
        return jsonify({'error': f"SageMaker Error: {str(e)}"}), 502
    except json.JSONDecodeError:
        return jsonify({'error': "Failed to decode model response"}), 502
    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app
    print(f"Starting Flask app acting as gateway to {ENDPOINT_NAME}...")
    app.run(host='0.0.0.0', port=5001, debug=True)
