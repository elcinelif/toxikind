#----------------------#
#       toxikind       #
#----------------------#
run_fit_save_feature_scaler:
	python -c "from toxikind.main import fit_save_feature_scaler; \
		fit_save_feature_scaler('raw_data/tox21_dense_train.csv.gz', 'production_model')"

run_load_transform_save_train_features:
	python -c "from toxikind.main import load_transform_save_features; \
		load_transform_save_features('production_model', 'raw_data/tox21_dense_train.csv.gz', 'data/X_train.csv')"

run_load_transform_save_test_features:
	python -c "from toxikind.main import load_transform_save_features; \
		load_transform_save_features('production_model', 'raw_data/tox21_dense_test.csv.gz', 'data/X_test.csv')"

run_preprocess_features: run_load_transform_save_train_features run_load_transform_save_test_features

run_load_save_train_targets:
	python -c "from toxikind.main import load_save_targets; \
		load_save_targets('raw_data/tox21_labels_train.csv.gz', 'data/y_train.csv')"

run_load_save_test_targets:
	python -c "from toxikind.main import load_save_targets; \
		load_save_targets('raw_data/tox21_labels_test.csv.gz', 'data/y_test.csv')"

run_preprocess_targets: run_load_save_train_targets run_load_save_test_targets

run_preprocess: run_preprocess_features run_preprocess_targets

run_model_load:

run_model_predict:

run_model_evaluate:

run_model: run_model_load run_model_predict run_model_evaluate

run_all: run_preprocess run_model

#======================#
# Install, clean, test #
#======================#

install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info

test_structure:
	@bash tests/test_structure.sh

#======================#
#          API         #
#======================#

run_api:
	uvicorn api.fast:app --reload --port 8000


#======================#
#          GCP         #
#======================#

gcloud-set-project:
	gcloud config set project $(GCP_PROJECT)



#======================#
#         Docker       #
#======================#

# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local \
		bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

DOCKER_IMAGE_PATH := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

docker_show_image_path:
	@echo $(DOCKER_IMAGE_PATH)

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

# Alternative if previous doesn´t work. Needs additional setup.
# Probably don´t need this. Used to build arm on linux amd64
docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod \
		bash

# Push and deploy to cloud

docker_allow:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev

docker_create_repo:
	gcloud artifacts repositories create $(DOCKER_REPO_NAME) \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="Repository for storing toxikind docker images"

docker_push:
	docker push $(DOCKER_IMAGE_PATH):prod

docker_deploy_with_yaml:
	gcloud run deploy \
		--image $(DOCKER_IMAGE_PATH):prod \
		--memory $(GAR_MEMORY) \
		--region $(GCP_REGION) \
		--env-vars-file .env.yaml

docker_deploy:
	gcloud run deploy \
		--image $(DOCKER_IMAGE_PATH):prod \
		--memory $(GAR_MEMORY) \
		--region $(GCP_REGION)
