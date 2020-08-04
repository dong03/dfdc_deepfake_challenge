#python3 predict.py --models trained/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36
#
#python3 predict.py --models trained/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19
#
#python3 predict.py --models trained/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29
#
#python3 predict.py --models trained/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31
#
#python3 predict.py --models trained/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37
#
#python3 predict.py --models trained/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40
#
#python3 predict.py --models trained/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23

GPU=$1

python3 predict.py --gpu $GPU \
 --models seed111_full_lr0.005DeepFakeClassifier_tf_efficientnet_b7_ns_0_best_dice \
 lr0.01_alliter_decayDeepFakeClassifier_tf_efficientnet_b7_ns_0_best_dice \
 full_lr0.005_alliterDeepFakeClassifier_tf_efficientnet_b7_ns_0_best_dice \
 classifier_DeepFakeClassifier_tf_efficientnet_b7_ns_0_best_dice\
 fix_lr0.01_decay0.8DeepFakeClassifier_tf_efficientnet_b7_ns_0_best_dice
# --models trained/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36 \
#  trained/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19 \
#  trained/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29 \
#  trained/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31 \
#  trained/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37 \
#  trained/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40 \
#  trained/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23