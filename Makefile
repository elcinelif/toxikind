#----------------------#
#       toxikind       #
#----------------------#
run_fit_save_feature_scaler:
	python toxikind/main.py run_fit_save_feature_scaler

run_load_transform_save_train_features:
	python toxikind/main.py run_load_transform_save_train_features

run_load_transform_save_test_features:
	python toxikind/main.py run_load_transform_save_test_features

run_load_transform_save_valid_features: #There is no validation data in this project! Should cause error.
	python toxikind/main.py run_load_transform_save_valid_features

run_preprocess_features: run_load_transform_save_train_features run_load_transform_save_test_features

run_load_transform_save_train_targets:
	python toxikind/main.py run_load_transform_save_train_targets

run_load_transform_save_test_targets:
	python toxikind/main.py run_load_transform_save_test_targets

run_load_transform_save_valid_targets: #There is no validation data in this project! Should cause error
	python toxikind/main.py run_load_transform_save_valid_targets

run_preprocess_targets: run_load_transform_save_train_targets run_load_transform_save_test_targets

run_preprocess: run_preprocess_features run_preprocess_targets

run_show_train_features:
	python toxikind/main.py run_show_train_features

run_show_test_features:
	python toxikind/main.py run_show_test_features

run_show_train_targets:
	python toxikind/main.py run_show_train_targets

run_show_test_targets:
	python toxikind/main.py run_show_test_targets

run_show_invalid_data_input:
	python toxikind/main.py run_show_invalid_data_input

run_model_train_save:
	python toxikind/main.py run_model_train_save $(assay)
	#Example Usage: run_model_train_save assay=NR.AhR

run_show_model_details:
	python toxikind/main.py run_show_model_details $(assay)

run_model_predict:

run_model_evaluate:



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
