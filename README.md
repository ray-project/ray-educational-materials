# Ray Educational Materials

© 2022, Anyscale Inc. All Rights Reserved

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Generic/ray_logo.png" width="30%" loading="lazy">

<a href="https://github.com/ray-project/ray-educational-materials"><img src="https://img.shields.io/github/stars/ray-project/ray-educational-materials?logo=Ray" alt="github-stars"></a>
<a href="https://forms.gle/9TSdDYUgxYs8SA9e8"><img src="https://img.shields.io/badge/Ray-Join%20Slack-blue" alt="join-ray-slack"></a>
<a href="https://discuss.ray.io/"><img src="https://img.shields.io/badge/Discuss-Ask%20Questions-blue" alt="discuss"></a>
<a href="https://twitter.com/raydistributed"><img src="https://img.shields.io/twitter/follow/raydistributed?label=Follow" alt="twitter"></a>

[![Introductory notebooks test](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-introductory-modules.yml/badge.svg?branch=main)](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-introductory-modules.yml)
[![Ray core notebooks test](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-ray-core.yml/badge.svg?branch=main)](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-ray-core.yml)
[![Semantic segmentation notebooks test](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-semantic-segmentation.yml/badge.svg?branch=main)](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-semantic-segmentation.yml)
[![Observability notebooks test](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-observability-modules.yml/badge.svg)](https://github.com/ray-project/ray-educational-materials/actions/workflows/scheduled-test-observability-modules.yml)

Welcome to a collection of education materials focused on [Ray](https://www.ray.io/), a distributed compute framework for scaling your Python and machine learning workloads from a laptop to a cluster.

## Recommended Learning Path

| Module                                                                                                                                                                                                    | Description                                                                                                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Overview of Ray](https://github.com/ray-project/ray-educational-materials/blob/main/Introductory_modules/Overview_of_Ray.ipynb)                                                                          | An Overview of Ray and entire Ray ecosystem.                                                                                                                                         |
| [Introduction to Ray AI Runtime](https://github.com/ray-project/ray-educational-materials/blob/main/Introductory_modules/Introduction_to_Ray_AI_Runtime.ipynb)                                            | An Overview of the Ray AI Runtime.                                                                                                                                                   |
| [Ray Core: Remote Functions as Tasks](https://github.com/ray-project/ray-educational-materials/blob/main/Ray_Core/Ray_Core_1_Remote_Functions.ipynb)                                                      | Learn how arbitrary functions to be executed asynchronously on separate Python workers.                                                                                              |
| [Ray Core: Remote Objects](https://github.com/ray-project/ray-educational-materials/blob/main/Ray_Core/Ray_Core_2_Remote_Objects.ipynb)                                                                   | Learn about objects that can be stored anywhere in a Ray cluster.                                                                                                                    |
| [Ray Core: Remote Classes as Actors, part 1](https://github.com/ray-project/ray-educational-materials/blob/main/Ray_Core/Ray_Core_3_Remote_Classes_part_1.ipynb)                                          | Work with stateful actors.                                                                                                                                                           |
| [Ray Core: Remote Classes as Actors, part 2](https://github.com/ray-project/ray-educational-materials/blob/main/Ray_Core/Ray_Core_4_Remote_Classes_part_2.ipynb)                                          | Learn "Tree of Actors" pattern.                                                                                                                                                      |
| [Scaling batch inference](https://github.com/ray-project/ray-educational-materials/blob/main/Computer_vision_workloads/Semantic_segmentation/Scaling_batch_inference.ipynb)                               | Learn about scaling batch inference in computer vision with Ray.                                                                                                                     |
| [Optional: Batch inference with Ray Datasets](https://github.com/ray-project/ray-educational-materials/blob/main/Computer_vision_workloads/Semantic_segmentation/Batch_inference_with_Ray_Datasets.ipynb) | Bonus content for scaling batch inference using Ray Datasets.                                                                                                                        |
| [Scaling model training](https://github.com/ray-project/ray-educational-materials/blob/main/Computer_vision_workloads/Semantic_segmentation/Scaling_model_training.ipynb)                                 | Learn about scaling model training in computer vision with Ray.                                                                                                                      |
| [Ray observability part 1](https://github.com/ray-project/ray-educational-materials/blob/main/Observability/Ray_observability_part_1.ipynb)                                                               | Introducing the Ray State API and Ray Dashboard UI as tools for observing the Ray cluster and applications.                                                                          |
| [LLM model fine-tuning and batch inference](https://github.com/ray-project/ray-educational-materials/blob/main/NLP_workloads/Text_generation/LLM_finetuning_and_batch_inference.ipynb)                    | Fine-tuning a Hugging Face Transformer (FLAN-T5) on the Alpaca dataset. Also includes distributed hyperparameter tuning and batch inference.                                         |
| [Multilingual chat with Ray Serve](https://github.com/ray-project/ray-educational-materials/blob/main/Ray_Serve/Multilingual_Chat_with_Ray_Serve_GPU.ipynb)                                               | Serving a Hugging Face LLM chat model with Ray Serve. Integrating multiple models and services within Ray Serve (language detection and translation) to implement multilingual chat. |
| [Ray Train featuring Tensorflow and word2vec](https://github.com/ray-project/ray-educational-materials/blob/main/NLP_workloads/Text_embeddings/Train_embeddings_tensorflow.ipynb)                                            | Intro to Ray Train. Distributed training with TensorflowTrainer. Porting a word2vec model from single-machine Tensorflow to distributed training with Ray. |

## Connect with the Ray community

You can learn and get more involved with the Ray community of developers and researchers:

* [**Ray documentation**](https://docs.ray.io/en/latest)

* [**Official Ray site**](https://www.ray.io/)
Browse the ecosystem and use this site as a hub to get the information that you need to get going and building with Ray.

* [**Join the community on Slack**](https://forms.gle/9TSdDYUgxYs8SA9e8)
Find friends to discuss your new learnings in our Slack space.

* [**Use the discussion board**](https://discuss.ray.io/)
Ask questions, follow topics, and view announcements on this community forum.

* [**Join a meetup group**](https://www.meetup.com/Bay-Area-Ray-Meetup/)
Tune in on meet-ups to listen to compelling talks, get to know other users, and meet the team behind Ray.

* [**Open an issue**](https://github.com/ray-project/ray/issues/new/choose)
Ray is constantly evolving to improve developer experience. Submit feature requests, bug-reports, and get help via GitHub issues.

* [**Become a Ray contributor**](https://docs.ray.io/en/latest/ray-contribute/getting-involved.html)
We welcome community contributions to improve our documentation and Ray framework.

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Generic/ray_logo.png" width="30%" loading="lazy">
