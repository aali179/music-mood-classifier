"�t
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@aоD��?iоD��?�Unknown�
BHostIDLE"IDLE1����Lf�@A����Lf�@awO�&�t�?i�$	s�?�Unknown
iHostWriteSummary"WriteSummary(1     Pa@9     Pa@A     Pa@I     Pa@a���~��m?iMŢ���?�Unknown�
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff�]@9fffff�]@A33333S[@I33333S[@a쓨�S[g?i�mvs��?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1������Q@9������Q@A�����LI@I�����LI@a	EZ16�U?i�3C��?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      B@9      B@A      B@I      B@aٖ���N?i*>Қ���?�Unknown�
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1������4@9������4@A33333�2@I33333�2@a:�M��??i�g����?�Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1ffffff)@9ffffff)@Affffff)@Iffffff)@aG��5?i��\���?�Unknown
�
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff+@9ffffff+@A333333&@I333333&@a�剔��2?i��H�	��?�Unknown
oHostSoftmax"model_5/MoodOutput/Softmax(1      $@9      $@A      $@I      $@a@p��s1?ilѾ�,��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������#@9������#@A������#@I������#@aIϐB��0?i�#G�D��?�Unknown
nHost_FusedMatMul"model_5/dense_5/Relu(1333333#@9333333#@A333333#@I333333#@aR.~�di0?iL���Q��?�Unknown
hHostRandomShuffle"RandomShuffle(1333333"@9333333"@A333333"@I333333"@a�7�{$/?i?M)�C��?�Unknown
}HostMatMul")gradient_tape/model_5/MoodOutput/MatMul_1(1������!@9������!@A������!@I������!@a��y�n.?i�$��*��?�Unknown
\HostSub"RMSprop/sub(1������@9������@A������@I������@aNj�h�S*?i��I����?�Unknown
^HostCast"model_5/Cast(1������@9������@A������@I������@a{E==�(?iY���Y��?�Unknown
�HostBiasAddGrad"4gradient_tape/model_5/MoodOutput/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a��*ΏF(?ie�*���?�Unknown
dHostDataset"Iterator::Model(1������3@9������3@A������@I������@a�b���'?iYeɢW��?�Unknown
xHostMatMul"$gradient_tape/model_5/dense_5/MatMul(1������@9������@A������@I������@a�b���'?i�e����?�Unknown
rHostTensorSliceDataset"TensorSliceDataset(1      @9      @A      @I      @a�޺3c9&?i]��4��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������@9������@A������@I������@a�=����%?i���Β��?�Unknown
{HostMatMul"'gradient_tape/model_5/MoodOutput/MatMul(1������@9������@A������@I������@a.�ȏ��!?il��F���?�Unknown
mHostMul"RMSprop/RMSprop/update_2/mul(1ffffff@9ffffff@Affffff@Iffffff@a7� �o!?i��F���?�Unknown
^HostGatherV2"GatherV2(1������@9������@A������@I������@aIϐB�� ?i�gU���?�Unknown
�HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1333333@9333333@A333333@I333333@aR.~�di ?i�T�����?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1������@9������@A������@I������@a[�kd� ?iv��	���?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a�ر�t?i�����?�Unknown
tHost_FusedMatMul"model_5/MoodOutput/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@a�ر�t?i�FIT���?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a�rp�?i{�����?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     a@9     a@A������@I������@a�rp�?ibhPۜ��?�Unknown
� HostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a ���a	?i]&}��?�Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @a3MҵRZ?i����W��?�Unknown
Z"HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@aWɇ�4�?i�'��?�Unknown
�#HostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@aWɇ�4�?i,OB����?�Unknown
X$HostEqual"Equal(1������@9������@A������@I������@ai�b&M?i@*s%���?�Unknown
w%HostReadVariableOp"div_no_nan_1/ReadVariableOp(1������@9������@A������@I������@ai�b&M?iT�����?�Unknown
k&HostMul"RMSprop/RMSprop/update/mul(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�͢�?i��A��?�Unknown
�'HostBiasAddGrad"1gradient_tape/model_5/dense_5/BiasAdd/BiasAddGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�͢�?i,2N����?�Unknown
�(HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1������	@9������	@A������	@I������	@a�=����?inW,����?�Unknown
m)HostMul"RMSprop/RMSprop/update_1/mul(1������	@9������	@A������	@I������	@a�=����?i�|
�S��?�Unknown
�*HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@a�w8*��?it΃`���?�Unknown
a+HostIdentity"Identity(1ffffff@9ffffff@Affffff@Iffffff@a
6L�%?i/�����?�Unknown�
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@a��m�v?i~�B��?�Unknown
�-HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a.�ȏ��?i�&~���?�Unknown
|.HostReluGrad"&gradient_tape/model_5/dense_5/ReluGrad(1������@9������@A������@I������@a.�ȏ��?i
�:�;��?�Unknown
�/HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1      @9      @A      @I      @a@p��s?i&(�}���?�Unknown
�0HostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1333333@9333333@A333333@I333333@aR.~�di?i���G��?�Unknown
�1HostReadVariableOp")model_5/MoodOutput/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�ر�t?i�n�����?�Unknown
X2HostCast"Cast_3(1������@9������@A������@I������@a�Tg.�?i{(��=��?�Unknown
s3HostSquare"RMSprop/RMSprop/update_3/Square(1������@9������@A������@I������@a�Tg.�?i�P���?�Unknown
X4HostSlice"Slice(1������@9������@A������@I������@a�Tg.�?i��X�.��?�Unknown
�5HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������ @9������ @A������ @I������ @a�rp�?i(d����?�Unknown
�6HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1       @9       @A       @I       @a3MҵRZ?iq;e���?�Unknown
�7HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1       @9       @A       @I       @a3MҵRZ?i��^|��?�Unknown
t8HostAssignAddVariableOp"AssignAddVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aWɇ�4�	?i���O���?�Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aWɇ�4�	?i��W@L��?�Unknown
�:HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1ffffff�?9ffffff�?Affffff�?Iffffff�?aWɇ�4�	?i�+1���?�Unknown
y;HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aWɇ�4�	?i6��!��?�Unknown
o<HostMul"RMSprop/RMSprop/update_2/mul_2(1333333�?9333333�?A333333�?I333333�?a����??i��!y��?�Unknown
b=HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a����??i̲�!���?�Unknown
o>HostMul"RMSprop/RMSprop/update_3/mul_2(1�������?9�������?A�������?I�������?a�=����?im�:�-��?�Unknown
�?HostReadVariableOp"(model_5/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�=����?iة0���?�Unknown
�@HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�=����?i������?�Unknown
�AHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1      �?9      �?A      �?I      �?a�]��?i&�.��?�Unknown
oBHostSqrt"RMSprop/RMSprop/update_2/Sqrt(1      �?9      �?A      �?I      �?a�]��?i�-	ր��?�Unknown
�CHostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%?i�]�l���?�Unknown
oDHostMul"RMSprop/RMSprop/update_1/mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%?i7���?�Unknown
�EHostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%?i����f��?�Unknown
uFHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%?i��0���?�Unknown
oGHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%?i�����?�Unknown
�HHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%?ikO]L��?�Unknown
�IHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1�������?9�������?A�������?I�������?a.�ȏ��?i��{���?�Unknown
yJHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1�������?9�������?A�������?I�������?a.�ȏ��?i��$����?�Unknown
oKHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1�������?9�������?A�������?I�������?a.�ȏ��?i�/�!��?�Unknown
oLHostAddV2"RMSprop/RMSprop/update_2/add(1�������?9�������?A�������?I�������?a.�ȏ��?i�K9�h��?�Unknown
�MHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a.�ȏ��?i�C���?�Unknown
XNHostCast"Cast_2(1333333�?9333333�?A333333�?I333333�?aR.~�di ?i�֘���?�Unknown
qOHostSquare"RMSprop/RMSprop/update/Square(1333333�?9333333�?A333333�?I333333�?aR.~�di ?i'j>3��?�Unknown
sPHostSquare"RMSprop/RMSprop/update_2/Square(1333333�?9333333�?A333333�?I333333�?aR.~�di ?iu��t��?�Unknown
�QHostReadVariableOp"%model_5/dense_5/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aR.~�di ?i�����?�Unknown
wRHostReadVariableOp"RMSprop/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�Tg.��>i������?�Unknown
qSHostAddV2"RMSprop/RMSprop/update_2/add_1(1�������?9�������?A�������?I�������?a�Tg.��>i�|��.��?�Unknown
oTHostMul"RMSprop/RMSprop/update_3/mul_1(1�������?9�������?A�������?I�������?a�Tg.��>ik��k��?�Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_1(1      �?9      �?A      �?I      �?a3MҵRZ�>iE�š��?�Unknown
TVHostMul"Mul(1      �?9      �?A      �?I      �?a3MҵRZ�>i��0z���?�Unknown
�WHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1      �?9      �?A      �?I      �?a3MҵRZ�>iZ�.��?�Unknown
�XHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1      �?9      �?A      �?I      �?a3MҵRZ�>i��{�E��?�Unknown
�YHostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1      �?9      �?A      �?I      �?a3MҵRZ�>i�� �|��?�Unknown
mZHostSqrt"RMSprop/RMSprop/update/Sqrt(1      �?9      �?A      �?I      �?a3MҵRZ�>iI_�L���?�Unknown
�[HostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1      �?9      �?A      �?I      �?a3MҵRZ�>i��k���?�Unknown
o\HostMul"RMSprop/RMSprop/update_2/mul_1(1      �?9      �?A      �?I      �?a3MҵRZ�>i�6� ��?�Unknown
q]HostAddV2"RMSprop/RMSprop/update_3/add_1(1      �?9      �?A      �?I      �?a3MҵRZ�>i8��jW��?�Unknown
`^HostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a3MҵRZ�>i�\���?�Unknown
v_HostCast"$sparse_categorical_crossentropy/Cast(1      �?9      �?A      �?I      �?a3MҵRZ�>i�y����?�Unknown
v`HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a{E==��>i��/���?�Unknown
maHostAddV2"RMSprop/RMSprop/update/add(1�������?9�������?A�������?I�������?a{E==��>ixn^L'��?�Unknown
mbHostMul"RMSprop/RMSprop/update/mul_1(1�������?9�������?A�������?I�������?a{E==��>i�茈X��?�Unknown
kcHostSub"RMSprop/RMSprop/update/sub(1�������?9�������?A�������?I�������?a{E==��>inc�ĉ��?�Unknown
odHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a{E==��>i��� ���?�Unknown
meHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a{E==��>idX=���?�Unknown
�fHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a{E==��>i��Fy��?�Unknown
�gHostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�=�����>i/\�<I��?�Unknown
shHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a�=�����>i� u��?�Unknown
miHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a�=�����>i�nmĠ��?�Unknown
�jHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?a�=�����>i�$����?�Unknown
okHostAddV2"RMSprop/RMSprop/update_3/add(1�������?9�������?A�������?I�������?a�=�����>io��K���?�Unknown
ulHostRealDiv" RMSprop/RMSprop/update_3/truediv(1�������?9�������?A�������?I�������?a�=�����>i�
�$��?�Unknown
�mHostReadVariableOp"&model_5/dense_5/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�=�����>i�K�O��?�Unknown
onHostAddV2"RMSprop/RMSprop/update/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%�>i5,�v��?�Unknown
moHostMul"RMSprop/RMSprop/update/mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%�>i[��i���?�Unknown
spHostRealDiv"RMSprop/RMSprop/update/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%�>i�\����?�Unknown
oqHostAddV2"RMSprop/RMSprop/update_1/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%�>i��M ���?�Unknown
mrHostSub"RMSprop/RMSprop/update_3/sub(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
6L�%�>i͌�K��?�Unknown
�sHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aR.~�di�>i�3X0��?�Unknown
�tHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aR.~�di�>i��!�P��?�Unknown
quHostAddV2"RMSprop/RMSprop/update_1/add_1(1333333�?9333333�?A333333�?I333333�?aR.~�di�>i����q��?�Unknown
�vHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aR.~�di�>i�(�����?�Unknown
ywHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?aR.~�di�>i��~i���?�Unknown
uxHostRealDiv" RMSprop/RMSprop/update_1/truediv(1      �?9      �?A      �?I      �?a3MҵRZ�>i�������?�Unknown
uyHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a3MҵRZ�>i];$���?�Unknown
wzHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�=�����>i     �?�Unknown*�r
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@a�`�^�_�?i�`�^�_�?�Unknown�
iHostWriteSummary"WriteSummary(1     Pa@9     Pa@A     Pa@I     Pa@a<ca��p?iY#�z���?�Unknown�
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff�]@9fffff�]@A33333S[@I33333S[@a�f>�j?i�a�i/��?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1������Q@9������Q@A�����LI@I�����LI@aLՠ��X?if�I���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      B@9      B@A      B@I      B@aB�P���Q?i����H��?�Unknown�
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1������4@9������4@A33333�2@I33333�2@ar��39B?i���ֵ�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff)@9ffffff)@Affffff)@Iffffff)@acM97��8?iC����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff+@9ffffff+@A333333&@I333333&@ab#�j^�5?ig��@���?�Unknown
o	HostSoftmax"model_5/MoodOutput/Softmax(1      $@9      $@A      $@I      $@a�!~�}3?i������?�Unknown
�
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������#@9������#@A������#@I������#@aC�$�3?i�L�(v��?�Unknown
nHost_FusedMatMul"model_5/dense_5/Relu(1333333#@9333333#@A333333#@I333333#@a� ��2?iȬ����?�Unknown
hHostRandomShuffle"RandomShuffle(1333333"@9333333"@A333333"@I333333"@a���v�1?i��u��?�Unknown
}HostMatMul")gradient_tape/model_5/MoodOutput/MatMul_1(1������!@9������!@A������!@I������!@a�鈑�X1?iA��/��?�Unknown
\HostSub"RMSprop/sub(1������@9������@A������@I������@aD�Q��.?iFQ���?�Unknown
^HostCast"model_5/Cast(1������@9������@A������@I������@az�0�,?i4N�����?�Unknown
�HostBiasAddGrad"4gradient_tape/model_5/MoodOutput/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a�t���+?iK�ѩ���?�Unknown
dHostDataset"Iterator::Model(1������3@9������3@A������@I������@aDj�#��*?ib�s:��?�Unknown
xHostMatMul"$gradient_tape/model_5/dense_5/MatMul(1������@9������@A������@I������@aDj�#��*?iy7[���?�Unknown
rHostTensorSliceDataset"TensorSliceDataset(1      @9      @A      @I      @aCU��`V)?i�"�}��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������@9������@A������@I������@aPd��(?i�P����?�Unknown
{HostMatMul"'gradient_tape/model_5/MoodOutput/MatMul(1������@9������@A������@I������@aA1E$?i�d+<Q��?�Unknown
mHostMul"RMSprop/RMSprop/update_2/mul(1ffffff@9ffffff@Affffff@Iffffff@a���O�#?i��(Q���?�Unknown
^HostGatherV2"GatherV2(1������@9������@A������@I������@aC�$�#?i�(�����?�Unknown
�HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1333333@9333333@A333333@I333333@a� ��"?i���K���?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1������@9������@A������@I������@a��pq&R"?i��Nn��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a���\�!?inT0��?�Unknown
tHost_FusedMatMul"model_5/MoodOutput/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@a���\�!?i��9O��?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a�ܠ�2_ ?i �,U��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     a@9     a@A������@I������@a�ܠ�2_ ?i. ( [��?�Unknown
�HostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a��!���?i;���Z��?�Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a��;/?iH��PT��?�Unknown
Z HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a����?iT?"QA��?�Unknown
�!HostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a����?i`��Q.��?�Unknown
X"HostEqual"Equal(1������@9������@A������@I������@a����}�?il���?�Unknown
w#HostReadVariableOp"div_no_nan_1/ReadVariableOp(1������@9������@A������@I������@a����}�?ix/�����?�Unknown
k$HostMul"RMSprop/RMSprop/update/mul(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�Z!+�?i�������?�Unknown
�%HostBiasAddGrad"1gradient_tape/model_5/dense_5/BiasAdd/BiasAddGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�Z!+�?i��F|���?�Unknown
�&HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1������	@9������	@A������	@I������	@aPd��?i���_��?�Unknown
m'HostMul"RMSprop/RMSprop/update_1/mul(1������	@9������	@A������	@I������	@aPd��?i�ᬥ&��?�Unknown
�(HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@a�0�J؛?i�6o����?�Unknown
a)HostIdentity"Identity(1ffffff@9ffffff@Affffff@Iffffff@a&��C�?i��&���?�Unknown�
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@a�a�?i��2��?�Unknown
�+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?aA1E?iɟԴ���?�Unknown
|,HostReluGrad"&gradient_tape/model_5/dense_5/ReluGrad(1������@9������@A������@I������@aA1E?i�)��v��?�Unknown
�-HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1      @9      @A      @I      @a�!~�}?i�����?�Unknown
�.HostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1333333@9333333@A333333@I333333@a� ��?i�rXy���?�Unknown
�/HostReadVariableOp")model_5/MoodOutput/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a���\�?i�19�7��?�Unknown
X0HostCast"Cast_3(1������@9������@A������@I������@a��d�&?i�Wt"���?�Unknown
s1HostSquare"RMSprop/RMSprop/update_3/Square(1������@9������@A������@I������@a��d�&?i�}�XJ��?�Unknown
X2HostSlice"Slice(1������@9������@A������@I������@a��d�&?i������?�Unknown
�3HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������ @9������ @A������ @I������ @a�ܠ�2_?i1��V��?�Unknown
�4HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1       @9       @A       @I       @a��;/?i%pE���?�Unknown
�5HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1       @9       @A       @I       @a��;/?i`P��?�Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a����?it�����?�Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a����?i ��=��?�Unknown
�8HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1ffffff�?9ffffff�?Affffff�?Iffffff�?a����?i&*?����?�Unknown
y9HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a����?i,��*��?�Unknown
o:HostMul"RMSprop/RMSprop/update_2/mul_2(1333333�?9333333�?A333333�?I333333�?aeAʿ�
?i2��
���?�Unknown
b;HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?aeAʿ�
?i8ׇ���?�Unknown
o<HostMul"RMSprop/RMSprop/update_3/mul_2(1�������?9�������?A�������?I�������?aPd��?i=g��a��?�Unknown
�=HostReadVariableOp"(model_5/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aPd��?iB�:����?�Unknown
�>HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?aPd��?iG��p)��?�Unknown
�?HostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1      �?9      �?A      �?I      �?a;��lc?iL~H����?�Unknown
o@HostSqrt"RMSprop/RMSprop/update_2/Sqrt(1      �?9      �?A      �?I      �?a;��lc?iQu�����?�Unknown
�AHostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C�?iV�
�;��?�Unknown
oBHostMul"RMSprop/RMSprop/update_1/mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C�?i[1.���?�Unknown
�CHostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C�?i`�'���?�Unknown
uDHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C�?ie�5�A��?�Unknown
oEHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C�?ijKD!���?�Unknown
�FHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C�?io�Rr���?�Unknown
�GHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1�������?9�������?A�������?I�������?aA1E?isn��A��?�Unknown
yHHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1�������?9�������?A�������?I�������?aA1E?iw3$����?�Unknown
oIHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1�������?9�������?A�������?I�������?aA1E?i{������?�Unknown
oJHostAddV2"RMSprop/RMSprop/update_2/add(1�������?9�������?A�������?I�������?aA1E?i���4��?�Unknown
�KHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1�������?9�������?A�������?I�������?aA1E?i��^؅��?�Unknown
XLHostCast"Cast_2(1333333�?9333333�?A333333�?I333333�?a� ��?i��!����?�Unknown
qMHostSquare"RMSprop/RMSprop/update/Square(1333333�?9333333�?A333333�?I333333�?a� ��?i�����?�Unknown
sNHostSquare"RMSprop/RMSprop/update_2/Square(1333333�?9333333�?A333333�?I333333�?a� ��?i��_f��?�Unknown
�OHostReadVariableOp"%model_5/dense_5/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a� ��?i�2k7���?�Unknown
wPHostReadVariableOp"RMSprop/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a��d�&?i�ň����?�Unknown
qQHostAddV2"RMSprop/RMSprop/update_2/add_1(1�������?9�������?A�������?I�������?a��d�&?i�X�m:��?�Unknown
oRHostMul"RMSprop/RMSprop/update_3/mul_1(1�������?9�������?A�������?I�������?a��d�&?i�����?�Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      �?9      �?A      �?I      �?a��;/�>i��;g���?�Unknown
TTHostMul"Mul(1      �?9      �?A      �?I      �?a��;/�>i�߳����?�Unknown
�UHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1      �?9      �?A      �?I      �?a��;/�>i��+$:��?�Unknown
�VHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1      �?9      �?A      �?I      �?a��;/�>i�ӣ�x��?�Unknown
�WHostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1      �?9      �?A      �?I      �?a��;/�>i�����?�Unknown
mXHostSqrt"RMSprop/RMSprop/update/Sqrt(1      �?9      �?A      �?I      �?a��;/�>i�Ǔ?���?�Unknown
�YHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1      �?9      �?A      �?I      �?a��;/�>i���3��?�Unknown
oZHostMul"RMSprop/RMSprop/update_2/mul_1(1      �?9      �?A      �?I      �?a��;/�>i����q��?�Unknown
q[HostAddV2"RMSprop/RMSprop/update_3/add_1(1      �?9      �?A      �?I      �?a��;/�>i���Z���?�Unknown
`\HostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a��;/�>i��s����?�Unknown
v]HostCast"$sparse_categorical_crossentropy/Cast(1      �?9      �?A      �?I      �?a��;/�>i���-��?�Unknown
v^HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?az�0��>i�
�9e��?�Unknown
m_HostAddV2"RMSprop/RMSprop/update/add(1�������?9�������?A�������?I�������?az�0��>i�k�[���?�Unknown
m`HostMul"RMSprop/RMSprop/update/mul_1(1�������?9�������?A�������?I�������?az�0��>i��b}���?�Unknown
kaHostSub"RMSprop/RMSprop/update/sub(1�������?9�������?A�������?I�������?az�0��>i�-5���?�Unknown
obHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?az�0��>iώ�E��?�Unknown
mcHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?az�0��>i����}��?�Unknown
�dHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?az�0��>i�P����?�Unknown
�eHostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1�������?9�������?A�������?I�������?aPd���>i������?�Unknown
sfHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?aPd���>i�����?�Unknown
mgHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?aPd���>iި2�K��?�Unknown
�hHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?aPd���>i�p_�}��?�Unknown
oiHostAddV2"RMSprop/RMSprop/update_3/add(1�������?9�������?A�������?I�������?aPd���>i�8�~���?�Unknown
ujHostRealDiv" RMSprop/RMSprop/update_3/truediv(1�������?9�������?A�������?I�������?aPd���>i� �c���?�Unknown
�kHostReadVariableOp"&model_5/dense_5/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?aPd���>i���H��?�Unknown
olHostAddV2"RMSprop/RMSprop/update/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C��>i��l�>��?�Unknown
mmHostMul"RMSprop/RMSprop/update/mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C��>i�&��j��?�Unknown
snHostRealDiv"RMSprop/RMSprop/update/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C��>i�U{B���?�Unknown
ooHostAddV2"RMSprop/RMSprop/update_1/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C��>i�����?�Unknown
mpHostSub"RMSprop/RMSprop/update_3/sub(1ffffff�?9ffffff�?Affffff�?Iffffff�?a&��C��>i�������?�Unknown
�qHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a� ���>i�Ik���?�Unknown
�rHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a� ���>i��Lk8��?�Unknown
qsHostAddV2"RMSprop/RMSprop/update_1/add_1(1333333�?9333333�?A333333�?I333333�?a� ���>i�u.�]��?�Unknown
�tHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a� ���>i�C���?�Unknown
yuHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a� ���>i�����?�Unknown
uvHostRealDiv" RMSprop/RMSprop/update_1/truediv(1      �?9      �?A      �?I      �?a��;/�>i �-����?�Unknown
uwHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a��;/�>i�i���?�Unknown
wxHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?aPd���>i     �?�Unknown2GPU