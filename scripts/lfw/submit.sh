
bsub < scripts/lfw/lfw_posthoc.sh
bsub < scripts/lfw/lfw_mcdrop.sh

bsub < scripts/lfw/lfw_online_fix.sh
bsub < scripts/lfw/lfw_online_pos.sh
bsub < scripts/lfw/lfw_online_full.sh

bsub < scripts/lfw/lfw_online_arccos_pos.sh
bsub < scripts/lfw/lfw_online_arccos_full.sh
