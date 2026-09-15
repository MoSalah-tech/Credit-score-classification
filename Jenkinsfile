pipeline {
    agent {
        docker {
            image 'python:3.11-slim-buster'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }

    environment {
        // Define environment variables for your project
        VENV_DIR = ".venv"
        PYTHON_BIN = "${env.VENV_DIR}/bin/python"
    }

    stages {
        stage('Checkout') {
            steps {
                // This checks out the code from your Git repository
                checkout scm
                // It's good practice to print the commit ID for traceability
                sh 'git log -1 --pretty=%B'
            }
        }

        stage('Setup Environment') {
            steps {
                sh """
                    python -m venv ${VENV_DIR}
                    ${PYTHON_BIN} -m pip install --upgrade pip
                    ${PYTHON_BIN} -m pip install -r requirements.txt
                """
            }
        }

        stage('Lint & Test') {
            steps {
                sh """
                    ${PYTHON_BIN} -m flake8 src/
                    ${PYTHON_BIN} -m pytest tests/
                """
            }
        }

        stage('Train Model') {
            steps {
                // This stage runs your training script
                sh "${PYTHON_BIN} -m src.models.train"
            }
        }

        stage('Evaluate & Archive Artifacts') {
            steps {
                // Run the evaluation script
                sh "${PYTHON_BIN} -m src.models.evaluate"
                // Archive the trained model and plots as artifacts
                archiveArtifacts artifacts: 'models/*.joblib', allowEmptyArchive: true
                archiveArtifacts artifacts: 'reports/figures/*.png', allowEmptyArchive: true
            }
        }
    }

    post {
        always {
            // Clean up the workspace after the pipeline run
            try{
            cleanWs()
            } catch (Exception e) {
                echo "Workspace cleanup failed: ${e.getMessage()}"
            }
        }
        success {
            echo 'Pipeline succeeded! Model trained and artifacts archived.'
        }
        failure {
            echo 'Pipeline failed. Check the logs for details.'
        }
    }
}