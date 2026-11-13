# Component provenance and licenses

- Native estimator: modified ORB-SLAM3, upstream UZ-SLAMLab/ORB_SLAM3; retain native/ORB_SLAM3/LICENSE (GPLv3) and original copyright notices.
- Learned stereo network and included serialized pretrained weights: NVIDIA Fast-FoundationStereo / FoundationStereo lineage. The frozen source pin recorded by the runtime is 476f4249561f7c79ca707326954f9255643412a6. Retain frontend/network/LICENSE.txt and model_card.md. The included license permits redistribution with notices and limits use to non-commercial research. These are upstream pretrained weights, not weights newly trained by the BHR-VIO authors.
- Pangolin: bundled source, native/Pangolin/LICENCE.
- DBoW2, g2o, Sophus: bundled under native/ORB_SLAM3/Thirdparty; retain their individual license/copyright files.

Licenses apply component by component. This package does not relicense the whole combined repository as MIT/Apache or grant commercial rights to third-party assets.
