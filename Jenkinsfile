pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "csvanalytics"
        DOCKER_COMPOSE_FILE = "docker-compose.yml"
        GIT_REPO = "https://github.com/priyabratakhandual/csvanalytics.git"
    }

    stages {
        stage('Clone Repository') {
            steps {
                git url: "${env.GIT_REPO}"
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh '/usr/bin/docker-compose build'
                }
            }
        }

        stage('Run Application') {
            steps {
                script {
                    sh '/usr/bin/docker-compose up -d'
                }
            }
        }

        stage('Optional Cleanup') {
            when {
                expression { return false } // change to true to enable cleanup
            }
            steps {
                script {
                    sh 'docker-compose down'
                }
            }
        }
    }
}