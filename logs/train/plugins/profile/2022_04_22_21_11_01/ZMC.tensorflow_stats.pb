"�k
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff���@9ffff���@Affff���@Iffff���@aQ�y����?iQ�y����?�Unknown�
BHostIDLE"IDLE133333{�@A33333{�@aH�����?iڔJM��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(13333337@93333337@A3333337@I3333337@ay ����C?i⿈[6��?�Unknown
iHostWriteSummary"WriteSummary(1     �4@9     �4@A     �4@I     �4@a��E�[A?iR*�H���?�Unknown�
pHostSoftmax"model_16/MoodOutput/Softmax(1      +@9      +@A      +@I      +@a�����6?iK�y�h��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      *@9      *@A      *@I      *@a����6?iM$�[)��?�Unknown
dHostDataset"Iterator::Model(1fffff�A@9fffff�A@A333333)@I333333)@a`�k1�V5?i�Q�+���?�Unknown
�	HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������0@9������0@A333333)@I333333)@a`�k1�V5?i/"�~��?�Unknown
p
Host_FusedMatMul"model_16/dense_16/Relu(1������(@9������(@A������(@I������(@a3=EY��4?iק����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333,@9333333,@A333333'@I333333'@ay ����3?i[�̔���?�Unknown
~HostMatMul"*gradient_tape/model_16/MoodOutput/MatMul_1(1������!@9������!@A������!@I������!@a�w(M*�-?i�qwp��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1       @9       @A       @I       @as>���+?iFO��!��?�Unknown
|HostMatMul"(gradient_tape/model_16/MoodOutput/MatMul(1      @9      @A      @I      @a��6S�f)?i��_o���?�Unknown
zHostMatMul"&gradient_tape/model_16/dense_16/MatMul(1      @9      @A      @I      @a�vws�'?i+*��3��?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1      @9      @A      @I      @a����&?i�E?���?�Unknown
qHostSquare"RMSprop/RMSprop/update/Square(1������@9������@A������@I������@a�ˑ	6�%?i�ޟ����?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1������@9������@A������@I������@a�ˑ	6�%?i�w �I��?�Unknown
mHostMul"RMSprop/RMSprop/update_2/mul(1ffffff@9ffffff@Affffff@Iffffff@a���$?iՉx=���?�Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a֮��bR$?i`�c���?�Unknown
mHostMul"RMSprop/RMSprop/update_1/mul(1������@9������@A������@I������@a�g�Э�#?i�&���?�Unknown
`HostGatherV2"
GatherV2_1(1333333@9333333@A333333@I333333@ay ����#?iH�nS��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@aKم DN#?i��QR���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a��%J"?i�:�����?�Unknown�
\HostSub"RMSprop/sub(1333333@9333333@A333333@I333333@a���p�!?i�9�+���?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@adu�络!?i	�u����?�Unknown
uHost_FusedMatMul"model_16/MoodOutput/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@a5.�F!?i��W���?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�A@933333�A@A������@I������@a�"^g�?i}�!���?�Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a��B<��?i���:���?�Unknown
�HostBiasAddGrad"5gradient_tape/model_16/MoodOutput/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a��B<��?i�d�i���?�Unknown
_HostCast"model_16/Cast(1ffffff@9ffffff@Affffff@Iffffff@a��B<��?i�F_����?�Unknown
� HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @as>���?iq�{]l��?�Unknown
w!HostReadVariableOp"RMSprop/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�!]+��?iZ�J:��?�Unknown
Z"HostArgMax"ArgMax(1������@9������@A������@I������@a]�{F?i��0���?�Unknown
�#HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������@9������@A������@I������@a���b?i�/����?�Unknown
m$HostMul"RMSprop/RMSprop/update/mul_2(1������@9������@A������@I������@a���b?i������?�Unknown
v%HostCast"$sparse_categorical_crossentropy/Cast(1������@9������@A������@I������@a���b?i?��L��?�Unknown
�&HostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1      @9      @A      @I      @a�vws�?i��|�	��?�Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a�_H��?i���y���?�Unknown
�(HostBiasAddGrad"3gradient_tape/model_16/dense_16/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a��%J?i��#�3��?�Unknown
�)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      �?A      @I      �?a�y7R�?i_o�E���?�Unknown
�*HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1333333@9333333@A333333@I333333@a�X-��A?iʨ�T=��?�Unknown
�+HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1������@9������@A������@I������@a�w(M*�?ilݢ����?�Unknown
`,HostDivNoNan"
div_no_nan(1������@9������@A������@I������@a�w(M*�?iL�+��?�Unknown
�-HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1������ @9������ @A������ @I������ @a,[��Vs?iKħ����?�Unknown
u.HostRealDiv" RMSprop/RMSprop/update_3/truediv(1������ @9������ @A������ @I������ @a,[��Vs?i�va��?�Unknown
~/HostReluGrad"(gradient_tape/model_16/dense_16/ReluGrad(1������ @9������ @A������ @I������ @a,[��Vs?i�(_.���?�Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @as>���?i�Xm����?�Unknown
�1HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1       @9       @A       @I       @as>���?iw�{�Y��?�Unknown
�2HostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1       @9       @A       @I       @as>���?iP��T���?�Unknown
�3HostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1       @9       @A       @I       @as>���?i)藶2��?�Unknown
s4HostSquare"RMSprop/RMSprop/update_3/Square(1�������?9�������?A�������?I�������?a���b?i9B���?�Unknown
b5HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a���b?iI>~����?�Unknown
�6HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a���b?iYi�XW��?�Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?aH�*j	?iy���?�Unknown
�8HostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aH�*j	?i��<���?�Unknown
X9HostCast"Cast_2(1�������?9�������?A�������?I�������?a�ˑ	6�?i��Nf��?�Unknown
X:HostEqual"Equal(1�������?9�������?A�������?I�������?a�ˑ	6�?i?����?�Unknown
X;HostCast"Cast_3(1      �?9      �?A      �?I      �?a֮��bR?i"�wL��?�Unknown
m<HostSqrt"RMSprop/RMSprop/update/Sqrt(1      �?9      �?A      �?I      �?a֮��bR?iO�_��?�Unknown
o=HostMul"RMSprop/RMSprop/update_1/mul_2(1      �?9      �?A      �?I      �?a֮��bR?i��߰��?�Unknown
o>HostSqrt"RMSprop/RMSprop/update_3/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H��?ifʽ���?�Unknown
�?HostReadVariableOp"*model_16/MoodOutput/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H��?i�5�H��?�Unknown
�@HostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1�������?9�������?A�������?I�������?adu�络?i������?�Unknown
�AHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?adu�络?it����?�Unknown
oBHostMul"RMSprop/RMSprop/update_3/mul_1(1�������?9�������?A�������?I�������?adu�络?i2����?�Unknown
�CHostReadVariableOp"'model_16/dense_16/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?adu�络?iL��gb��?�Unknown
oDHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1333333�?9333333�?A333333�?I333333�?a�X-��A ?i�go���?�Unknown
qEHostAddV2"RMSprop/RMSprop/update_1/add_1(1333333�?9333333�?A333333�?I333333�?a�X-��A ?i��	w���?�Unknown
oFHostMul"RMSprop/RMSprop/update_1/mul_1(1333333�?9333333�?A333333�?I333333�?a�X-��A ?ik�~%��?�Unknown
�GHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�X-��A ?i %N�f��?�Unknown
oHHostMul"RMSprop/RMSprop/update_3/mul_2(1333333�?9333333�?A333333�?I333333�?a�X-��A ?i�A�����?�Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�X-��A ?i�^�����?�Unknown
tJHostAssignAddVariableOp"AssignAddVariableOp(1�������?9�������?A�������?I�������?a�w(M*��>i���1$��?�Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a�w(M*��>i,�;�_��?�Unknown
�LHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1�������?9�������?A�������?I�������?a�w(M*��>i}-�j���?�Unknown
�MHostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a�w(M*��>i������?�Unknown
�NHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1�������?9�������?A�������?I�������?a�w(M*��>ib9���?�Unknown
oOHostSqrt"RMSprop/RMSprop/update_2/Sqrt(1�������?9�������?A�������?I�������?a�w(M*��>ip��?N��?�Unknown
�PHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?a�w(M*��>i���ۉ��?�Unknown
TQHostMul"Mul(1      �?9      �?A      �?I      �?as>����>i������?�Unknown
yRHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1      �?9      �?A      �?I      �?as>����>i���=���?�Unknown
sSHostSquare"RMSprop/RMSprop/update_2/Square(1      �?9      �?A      �?I      �?as>����>i���n,��?�Unknown
�THostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?as>����>iq���b��?�Unknown
qUHostAddV2"RMSprop/RMSprop/update_3/add_1(1      �?9      �?A      �?I      �?as>����>i]ј��?�Unknown
mVHostMul"RMSprop/RMSprop/update/mul_1(1�������?9�������?A�������?I�������?a���b�>i壿����?�Unknown
sWHostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?a���b�>im9y\���?�Unknown
�XHostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1�������?9�������?A�������?I�������?a���b�>i��2"+��?�Unknown
sYHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a���b�>i}d��[��?�Unknown
�ZHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a���b�>i������?�Unknown
o[HostAddV2"RMSprop/RMSprop/update_3/add(1�������?9�������?A�������?I�������?a���b�>i��_s���?�Unknown
m\HostAddV2"RMSprop/RMSprop/update/add(1�������?9�������?A�������?I�������?a�ˑ	6��>i�������?�Unknown
m]HostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a�ˑ	6��>iյ7(��?�Unknown
�^HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a�ˑ	6��>i�ȣ�?��?�Unknown
o_HostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?a�ˑ	6��>i��j��?�Unknown
m`HostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a�ˑ	6��>iA�{7���?�Unknown
maHostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?a�ˑ	6��>ie����?�Unknown
�bHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�ˑ	6��>i�T����?�Unknown
kcHostSub"RMSprop/RMSprop/update/sub(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H���>iH�r���?�Unknown
odHostAddV2"RMSprop/RMSprop/update_1/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H���>i7��8��?�Unknown
ueHostRealDiv" RMSprop/RMSprop/update_1/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H���>i�ǯ�^��?�Unknown
qfHostAddV2"RMSprop/RMSprop/update_2/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H���>i�XΨ���?�Unknown
ugHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H���>iD�엪��?�Unknown
�hHostReadVariableOp")model_16/MoodOutput/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�_H���>iz����?�Unknown
aiHostIdentity"Identity(1333333�?9333333�?A333333�?I333333�?a�X-��A�>i^��
���?�Unknown�
yjHostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�X-��A�>i������?�Unknown
okHostAddV2"RMSprop/RMSprop/update/add_1(1333333�?9333333�?A333333�?I333333�?a�X-��A�>i�~2��?�Unknown
�lHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�X-��A�>io�O�R��?�Unknown
omHostAddV2"RMSprop/RMSprop/update_2/add(1333333�?9333333�?A333333�?I333333�?a�X-��A�>i�� s��?�Unknown
onHostMul"RMSprop/RMSprop/update_2/mul_2(1333333�?9333333�?A333333�?I333333�?a�X-��A�>i%����?�Unknown
woHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�X-��A�>i���!���?�Unknown
�pHostReadVariableOp"(model_16/dense_16/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�X-��A�>i�쓥���?�Unknown
wqHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�ˑ	6��>im��R���?�Unknown
yrHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�ˑ	6��>i�������?�Unknown*�j
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff���@9ffff���@Affff���@Iffff���@a�ͭm��?i�ͭm��?�Unknown�
sHostDataset"Iterator::Model::ParallelMapV2(13333337@93333337@A3333337@I3333337@a�*l��E?i��h�ٱ�?�Unknown
iHostWriteSummary"WriteSummary(1     �4@9     �4@A     �4@I     �4@a�ζK�)C?iS�;$���?�Unknown�
pHostSoftmax"model_16/MoodOutput/Softmax(1      +@9      +@A      +@I      +@aa�D_=9?i_$�˹�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      *@9      *@A      *@I      *@a�J&�N8?i(w �ռ�?�Unknown
dHostDataset"Iterator::Model(1fffff�A@9fffff�A@A333333)@I333333)@a�8�.��7?i�R�eǿ�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������0@9������0@A333333)@I333333)@a�8�.��7?i�.l9���?�Unknown
pHost_FusedMatMul"model_16/dense_16/Relu(1������(@9������(@A������(@I������(@a�/���.7?i������?�Unknown
�	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333,@9333333,@A333333'@I333333'@a�*l��5?i=K�U��?�Unknown
~
HostMatMul"*gradient_tape/model_16/MoodOutput/MatMul_1(1������!@9������!@A������!@I������!@aN�1~�s0?io$�c��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1       @9       @A       @I       @a^�B+��-?i���/B��?�Unknown
|HostMatMul"(gradient_tape/model_16/MoodOutput/MatMul(1      @9      @A      @I      @a8��hM,?i�N����?�Unknown
zHostMatMul"&gradient_tape/model_16/dense_16/MatMul(1      @9      @A      @I      @axڥ�,*?i.������?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1      @9      @A      @I      @a�J&�N(?i�ޥ�*��?�Unknown
qHostSquare"RMSprop/RMSprop/update/Square(1������@9������@A������@I������@a�A�V�'?i�nv���?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1������@9������@A������@I������@a�A�V�'?i��v[(��?�Unknown
mHostMul"RMSprop/RMSprop/update_2/mul(1ffffff@9ffffff@Affffff@Iffffff@a�&�z*�&?i=�N���?�Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a�r qo&?i_�0E���?�Unknown
mHostMul"RMSprop/RMSprop/update_1/mul(1������@9������@A������@I������@a�NƷ&?i@�@]��?�Unknown
`HostGatherV2"
GatherV2_1(1333333@9333333@A333333@I333333@a�*l��%?i�֓@���?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a�EP%?iA��D��?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a��1$?i�0uVP��?�Unknown�
\HostSub"RMSprop/sub(1333333@9333333@A333333@I333333@a��u�_�#?i=�ol���?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a��QO�q#?iZ�Ԇ���?�Unknown
uHost_FusedMatMul"model_16/MoodOutput/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@a��-��#?i7�����?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�A@933333�A@A������@I������@ad����!?i����?�Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@am��^�?ii�!��?�Unknown
�HostBiasAddGrad"5gradient_tape/model_16/MoodOutput/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@am��^�?i��
l���?�Unknown
_HostCast"model_16/Cast(1ffffff@9ffffff@Affffff@Iffffff@am��^�?i�����?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a^�B+��?i.)c���?�Unknown
wHostReadVariableOp"RMSprop/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?���k?i�>�^���?�Unknown
Z HostArgMax"ArgMax(1������@9������@A������@I������@a1�j��?i�9����?�Unknown
�!HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������@9������@A������@I������@a"�"Z!�?i,�Dv��?�Unknown
m"HostMul"RMSprop/RMSprop/update/mul_2(1������@9������@A������@I������@a"�"Z!�?i@TO}M��?�Unknown
v#HostCast"$sparse_categorical_crossentropy/Cast(1������@9������@A������@I������@a"�"Z!�?iT%Z�$��?�Unknown
�$HostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1      @9      @A      @I      @axڥ�,?i(T�C���?�Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a��᷋�?i8-ȝ��?�Unknown
�&HostBiasAddGrad"3gradient_tape/model_16/dense_16/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a��1?i0�P?��?�Unknown
�'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      �?A      @I      �?a{�	�3�?iU�����?�Unknown
�(HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1333333@9333333@A333333@I333333@al�����?ic>�xd��?�Unknown
�)HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1������@9������@A������@I������@aN�1~�s?i�/u���?�Unknown
`*HostDivNoNan"
div_no_nan(1������@9������@A������@I������@aN�1~�s?i{!Q�k��?�Unknown
�+HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1������ @9������ @A������ @I������ @a}�ғ�h?i�p�Y���?�Unknown
u,HostRealDiv" RMSprop/RMSprop/update_3/truediv(1������ @9������ @A������ @I������ @a}�ғ�h?i���f��?�Unknown
~-HostReluGrad"(gradient_tape/model_16/dense_16/ReluGrad(1������ @9������ @A������ @I������ @a}�ғ�h?i_$����?�Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a^�B+��?ij��G\��?�Unknown
�/HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1       @9       @A       @I       @a^�B+��?iui�����?�Unknown
�0HostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1       @9       @A       @I       @a^�B+��?i�6�K��?�Unknown
�1HostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1       @9       @A       @I       @a^�B+��?i���>���?�Unknown
s2HostSquare"RMSprop/RMSprop/update_3/Square(1�������?9�������?A�������?I�������?a"�"Z!�
?i,l�.��?�Unknown
b3HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a"�"Z!�
?i�����?�Unknown
�4HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a"�"Z!�
?i)�vP��?�Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?af��;m	?is�fl��?�Unknown
�6HostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?af��;m	?i��V����?�Unknown
X7HostCast"Cast_2(1�������?9�������?A�������?I�������?a�A�V�?iƭ�s1��?�Unknown
X8HostEqual"Equal(1�������?9�������?A�������?I�������?a�A�V�?i��
-���?�Unknown
X9HostCast"Cast_3(1      �?9      �?A      �?I      �?a�r qo?i�S�����?�Unknown
m:HostSqrt"RMSprop/RMSprop/update/Sqrt(1      �?9      �?A      �?I      �?a�r qo?i_Փ�D��?�Unknown
o;HostMul"RMSprop/RMSprop/update_1/mul_2(1      �?9      �?A      �?I      �?a�r qo?i'WXf���?�Unknown
o<HostSqrt"RMSprop/RMSprop/update_3/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋�?i�6�(���?�Unknown
�=HostReadVariableOp"*model_16/MoodOutput/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋�?i7��E��?�Unknown
�>HostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1�������?9�������?A�������?I�������?a��QO�q?i~SO����?�Unknown
�?HostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a��QO�q?iŐ�w���?�Unknown
o@HostMul"RMSprop/RMSprop/update_3/mul_1(1�������?9�������?A�������?I�������?a��QO�q?i΁>/��?�Unknown
�AHostReadVariableOp"'model_16/dense_16/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��QO�q?iS}��?�Unknown
oBHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1333333�?9333333�?A333333�?I333333�?al�����?iZ�����?�Unknown
qCHostAddV2"RMSprop/RMSprop/update_1/add_1(1333333�?9333333�?A333333�?I333333�?al�����?iaA"���?�Unknown
oDHostMul"RMSprop/RMSprop/update_1/mul_1(1333333�?9333333�?A333333�?I333333�?al�����?ih�%fT��?�Unknown
�EHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?al�����?iow)1���?�Unknown
oFHostMul"RMSprop/RMSprop/update_3/mul_2(1333333�?9333333�?A333333�?I333333�?al�����?iv-����?�Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?al�����?i}�0�+��?�Unknown
tHHostAssignAddVariableOp"AssignAddVariableOp(1�������?9�������?A�������?I�������?aN�1~�s ?iC���m��?�Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?aN�1~�s ?i	�f���?�Unknown
�JHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1�������?9�������?A�������?I�������?aN�1~�s ?iϗz5���?�Unknown
�KHostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1�������?9�������?A�������?I�������?aN�1~�s ?i���3��?�Unknown
�LHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1�������?9�������?A�������?I�������?aN�1~�s ?i[�V�t��?�Unknown
oMHostSqrt"RMSprop/RMSprop/update_2/Sqrt(1�������?9�������?A�������?I�������?aN�1~�s ?i!�ģ���?�Unknown
�NHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?aN�1~�s ?i�z2s���?�Unknown
TOHostMul"Mul(1      �?9      �?A      �?I      �?a^�B+���>im�
G4��?�Unknown
yPHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1      �?9      �?A      �?I      �?a^�B+���>i�'�p��?�Unknown
sQHostSquare"RMSprop/RMSprop/update_2/Square(1      �?9      �?A      �?I      �?a^�B+���>iy~����?�Unknown
�RHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?a^�B+���>i�ԓ����?�Unknown
qSHostAddV2"RMSprop/RMSprop/update_3/add_1(1      �?9      �?A      �?I      �?a^�B+���>i�+l�#��?�Unknown
mTHostMul"RMSprop/RMSprop/update/mul_1(1�������?9�������?A�������?I�������?a"�"Z!��>i�߮nY��?�Unknown
sUHostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?a"�"Z!��>i��F���?�Unknown
�VHostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1�������?9�������?A�������?I�������?a"�"Z!��>iTH4���?�Unknown
sWHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a"�"Z!��>i��v����?�Unknown
�XHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a"�"Z!��>iް��0��?�Unknown
oYHostAddV2"RMSprop/RMSprop/update_3/add(1�������?9�������?A�������?I�������?a"�"Z!��>i#e��f��?�Unknown
mZHostAddV2"RMSprop/RMSprop/update/add(1�������?9�������?A�������?I�������?a�A�V��>i(w�����?�Unknown
m[HostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a�A�V��>i-�Va���?�Unknown
�\HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a�A�V��>i2�>���?�Unknown
o]HostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?a�A�V��>i7��&��?�Unknown
m^HostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a�A�V��>i<�]�U��?�Unknown
m_HostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?a�A�V��>iA�
ԅ��?�Unknown
�`HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�A�V��>iF㷰���?�Unknown
kaHostSub"RMSprop/RMSprop/update/sub(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋��>i
Sϑ���?�Unknown
obHostAddV2"RMSprop/RMSprop/update_1/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋��>i���r	��?�Unknown
ucHostRealDiv" RMSprop/RMSprop/update_1/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋��>i�2�S3��?�Unknown
qdHostAddV2"RMSprop/RMSprop/update_2/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋��>iV�5]��?�Unknown
ueHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋��>i-���?�Unknown
�fHostReadVariableOp")model_16/MoodOutput/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��᷋��>iށD����?�Unknown
agHostIdentity"Identity(1333333�?9333333�?A333333�?I333333�?al������>iaO�����?�Unknown�
yhHostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?al������>i�H����?�Unknown
oiHostAddV2"RMSprop/RMSprop/update/add_1(1333333�?9333333�?A333333�?I333333�?al������>ig�ɧ��?�Unknown
�jHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?al������>i�K�@��?�Unknown
okHostAddV2"RMSprop/RMSprop/update_2/add(1333333�?9333333�?A333333�?I333333�?al������>im��rd��?�Unknown
olHostMul"RMSprop/RMSprop/update_2/mul_2(1333333�?9333333�?A333333�?I333333�?al������>i�ROX���?�Unknown
wmHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?al������>is �=���?�Unknown
�nHostReadVariableOp"(model_16/dense_16/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?al������>i��R#���?�Unknown
woHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�A�V��>i�v����?�Unknown
ypHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�A�V��>i�������?�Unknown2GPU