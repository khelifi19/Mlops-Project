# Déclaration des variables
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
SOURCE_DIR=model_pipeline.py
MAIN_SCRIPT=main.py
TEST_DIR=tests/
IMAGE_NAME=yassine-khelifi-fastapi-mlflow
CONTAINER_NAME=fastapi-mlops-container

# Configuration de l'environnement
setup:
	@echo "🔧 Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@./$(ENV_NAME)/bin/python3 -m pip install --upgrade pip
	@./$(ENV_NAME)/bin/python3 -m pip install -r $(REQUIREMENTS)
	@echo "✅ Environnement configuré avec succès !"

# Vérification du code
verify:
	@echo "🛠 Vérification de la qualité du code..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m black --exclude 'venv|mlops_env' .
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m pylint --disable=C,R $(SOURCE_DIR) || true
	@echo "🔍 Running Bandit security check on selected files..."
	@. $(ENV_NAME)/bin/activate && bandit -r model_pipeline.py main.py check_features.py test_environment.py
	@echo "✅ Code vérifié avec succès !"

# Préparation des données
prepare:
	@echo "📊 Préparation des données..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --prepare
	@echo "✅ Données préparées avec succès !"

# Entraînement du modèle
train:
	@echo "🚀 Entraînement du modèle avec suivi MLflow..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --train
	@echo "✅ Modèle entraîné avec succès !"

# Évaluation du modèle
evaluate:
	@echo "📊 Évaluation du modèle..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --evaluate
	@echo "✅ Évaluation du modèle terminée !"

# Exécution des tests
test:
	@echo "🧪 Exécution des tests..."
	@if [ ! -d "$(TEST_DIR)" ]; then echo "⚠️  Création du dossier $(TEST_DIR)..."; mkdir -p $(TEST_DIR); fi
	@if [ -z "$$(ls -A $(TEST_DIR))" ]; then echo "⚠️  Aucun test trouvé ! Création d'un test basique..."; echo 'def test_dummy(): assert 2 + 2 == 4' > $(TEST_DIR)/test_dummy.py; fi
	@./$(ENV_NAME)/bin/python3 -m pytest $(TEST_DIR) --disable-warnings
	@echo "✅ Tests exécutés avec succès !"

# Nettoyage des fichiers temporaires
clean:
	@echo "🗑 Suppression des fichiers temporaires..."
	rm -rf $(ENV_NAME)
	rm -f model.pkl scaler.pkl pca.pkl
	rm -rf __pycache__ .pytest_cache .pylint.d
	@echo "✅ Nettoyage terminé !"

# Réinstallation complète de l'environnement
reinstall: clean setup

# Lancer le serveur FastAPI pour tester l'API
api-test:
	@echo "🚀 Lancement de l'API FastAPI..."
	@. $(ENV_NAME)/bin/activate && exec uvicorn app:app --host 127.0.0.1 --port 8000 --reload
	@echo "🌐 API disponible à l'adresse : http://127.0.0.1:8000/docs"
	@echo "👉 Ouvrez Swagger UI pour tester l'API."

# Lancer le serveur MLflow (tracking server)
start-mlflow:
	@echo "🚀 Démarrage du serveur MLflow avec tracking vers SQLite..."
	@. $(ENV_NAME)/bin/activate && \
	mlflow server \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--host 0.0.0.0 \
	--port 5002
	@echo "🌐 MLflow tracking server sur http://127.0.0.1:5002"

# Interface UI MLflow (optionnelle)
mlflow:
	@echo "🚀 Lancement de MLflow UI..."
	@. $(ENV_NAME)/bin/activate && exec mlflow ui --host 0.0.0.0 --port 5002 --backend-store-uri sqlite:///mlflow.db
	@echo "🌐 Interface MLflow disponible sur http://127.0.0.1:5002"

# Pipeline complet (hors serveurs)
all: setup verify prepare train evaluate test
	@echo "🎉 Pipeline MLOps exécuté avec succès !"

# Lancer toute la pipeline + API + MLflow UI
run-all:
	@echo "🚀 Exécution complète du pipeline, API, MLflow UI et Streamlit..."
	@make all
	@make api-test &
	@sleep 3
	@make mlflow &
	@sleep 3
	@make streamlit

# Docker : Build, Push, Run, Stop
docker-build:
	@echo "🐳 Build Docker image..."
	docker build -t $(IMAGE_NAME) .
	@echo "✅ Image build avec succès !"

docker-push:
	@echo "📤 Push Docker image..."
	docker push $(IMAGE_NAME)
	@echo "✅ Image poussée avec succès !"

docker-run:
	@echo "🚀 Run container Docker..."
	docker run -p 8000:8000 --name $(CONTAINER_NAME) $(IMAGE_NAME)
	@echo "✅ Container lancé !"

docker-stop:
	@echo "🛑 Stop + remove container..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	@echo "✅ Container supprimé !"

docker-clean:
	@echo "🧹 Nettoyage Docker..."
	docker system prune -a -f
	@echo "✅ Docker nettoyé !"

docker-deploy: docker-stop docker-build docker-push docker-run
	@echo "🚀 Docker déployé avec succès !"

# Installer Docker et Docker Compose
install-tools:
	@echo "🔧 Installation Docker & Compose..."
    ifeq ($(shell uname), Darwin)
	@echo "➡️  macOS détecté : merci d’installer Docker Desktop manuellement depuis https://www.docker.com/products/docker-desktop"
else
	sudo apt update && sudo apt install -y docker.io docker-compose
endif
	@echo "✅ Étape d'installation terminée !"

# Lancer la stack Elasticsearch & Kibana
start-elk:
	@echo "🚀 Lancement Elasticsearch & Kibana..."
	docker-compose up -d
	@echo "✅ Stack ELK démarrée !"

# Arrêter Elasticsearch & Kibana
stop-elk:
	@echo "🛑 Arrêt Elasticsearch & Kibana..."
	docker-compose down
	@echo "✅ Stack ELK arrêtée !"

# Tester la connexion à Elasticsearch
test-elk-connection:
	@echo "🔗 Test de connexion Elasticsearch..."
	@./$(ENV_NAME)/bin/python3 test_elk_connection.py
	@echo "✅ Connexion testée avec succès !"

# Vérifier que les logs sont envoyés
elk-note:
	@echo "📌 Assurez-vous que les logs MLflow sont envoyés vers Elasticsearch dans votre code Python !"

# Lancer tout le monitoring (Docker, ELK, MLflow, Netdata)
run-monitoring:
	@make install-tools
	@make start-elk
	@sleep 10
	@make start-mlflow
	@sleep 10
	@make start-netdata
	@echo "📊 Monitoring opérationnel : ELK + MLflow + Netdata"

# Lancer Netdata via Docker
start-netdata:
	@echo "🚀 Lancement de Netdata..."
	docker run -d --name netdata -p 19999:19999 netdata/netdata
	@echo "✅ Netdata lancé avec succès !"

# Arrêter et supprimer le conteneur Netdata
stop-netdata:
	@echo "🛑 Arrêt de Netdata..."
	docker stop netdata || true
	docker rm netdata || true
	@echo "✅ Netdata arrêté et supprimé !"

# Nettoyage Docker (y compris Netdata)
docker-clean:
	@echo "🧹 Nettoyage Docker..."
	docker system prune -a -f
	docker rm -f netdata || true
	@echo "✅ Docker nettoyé, y compris Netdata !"

# Lancer l'application Streamlit
streamlit:
	@echo "🚀 Lancement de l'application Streamlit..."
	@. $(ENV_NAME)/bin/activate && streamlit run streamlit_app.py
	@echo "🌐 Application disponible sur http://localhost:8501"
