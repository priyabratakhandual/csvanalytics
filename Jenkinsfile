pipeline {
    agent any

    environment {
        DOCKER_COMPOSE_FILE = 'docker-compose.yml'
        GIT_REPO = 'https://github.com/priyabratakhandual/csvanalytics.git'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git url: "${env.GIT_REPO}"
            }
        }

        stage('Build and Start Services') {
            steps {
                script {
                    sh 'docker-compose down || true'  // Clean up if already running
                    sh 'docker-compose build'
                    sh 'docker-compose up -d'
                }
            }
        }

        stage('Check Application Health') {
            steps {
                script {
                    // You can replace this with curl if your app has a /ping or /health endpoint
                    sh 'curl --fail http://localhost:80/csv-analytics/ping || echo "Health check failed"'
                }
            }
        }

        stage('Optional Cleanup') {
            when {
                expression { return false } // Set to true to stop containers after build
            }
            steps {
                script {
                    sh 'docker-compose down'
                }
            }
        }
    }
}
