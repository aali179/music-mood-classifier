"�t
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1    ���@9    ���@A    ���@I    ���@a�`,����?i�`,����?�Unknown�
BHostIDLE"IDLE1�����N�@A�����N�@a}��8 ��?i��D����?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1�����Lb@9�����Lb@Afffffa@Ifffffa@ap��m?i� 7��?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(133333�Q@933333�Q@A������E@I������E@a�0�EhR?ip�YD��?�Unknown
iHostWriteSummary"WriteSummary(1������9@9������9@A������9@I������9@a����E?iv��|í�?�Unknown�
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1�����;@9�����;@A      9@I      9@a�'s�NE?i@I���?�Unknown
�HostBiasAddGrad"4gradient_tape/model_4/MoodOutput/BiasAdd/BiasAddGrad(1�����4@9�����4@A�����4@I�����4@a�p��!A?iܾ@_��?�Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1������3@9������3@A������3@I������3@aY3��@?i"]%���?�Unknown
o
HostSoftmax"model_4/MoodOutput/Softmax(1fffff�0@9fffff�0@Afffff�0@Ifffff�0@ak�Q��<?iϻ�0��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������(@9������(@A������(@I������(@a���b"5?imXf,���?�Unknown
}HostMatMul")gradient_tape/model_4/MoodOutput/MatMul_1(1333333(@9333333(@A333333(@I333333(@a�8:w}�4?i�?i��?�Unknown
nHost_FusedMatMul"model_4/dense_4/Relu(1333333'@9333333'@A333333'@I333333'@amsPT�3?iN�����?�Unknown
xHostMatMul"$gradient_tape/model_4/dense_4/MatMul(1333333$@9333333$@A333333$@I333333$@aŎ��61?i�Ѻ���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      #@9      #@A      #@I      #@a�(��10?i��v���?�Unknown�
dHostDataset"Iterator::Model(1�����=@9�����=@A������"@I������"@a,bWo��/?iC��	��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������!@9������!@A������!@I������!@a|��ʅV.?i��e���?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff%@9ffffff%@Affffff @Iffffff @a�@+��+?i@ݟ���?�Unknown
hHostRandomShuffle"RandomShuffle(1333333 @9333333 @A333333 @I333333 @a��h�+?i�`efh��?�Unknown
oHostAddV2"RMSprop/RMSprop/update_2/add(1       @9       @A       @I       @a�Q��$E+?i���?�Unknown
rHostTensorSliceDataset"TensorSliceDataset(1ffffff@9ffffff@Affffff@Iffffff@a��V�q&?i^�̯���?�Unknown
{HostMatMul"'gradient_tape/model_4/MoodOutput/MatMul(1ffffff@9ffffff@Affffff@Iffffff@a��V�q&?i�����?�Unknown
�HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1333333@9333333@A333333@I333333@a�c��y%?i��OAD��?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1      @9      @A      @I      @aE����s$?i�$
���?�Unknown
\HostSub"RMSprop/sub(1      @9      @A      @I      @aE����s$?iaOļ���?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1ffffff@9ffffff@Affffff@Iffffff@a�:��#?i3�)��?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@aR��"?iOa�9%��?�Unknown
ZHostArgMax"ArgMax(1������@9������@A������@I������@a��r^� ?i}H�x0��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     @d@9     @d@A������@I������@a��r^� ?i�/ �;��?�Unknown
�HostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1������@9������@A������@I������@a�!B�?i�=1�+��?�Unknown
tHost_FusedMatMul"model_4/MoodOutput/BiasAdd(1333333@9333333@A333333@I333333@a8�ϺP?it�8��?�Unknown
` HostGatherV2"
GatherV2_1(1������@9������@A������@I������@aa/W}3�?i-��I���?�Unknown
^!HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a�@+��?i�����?�Unknown
l"HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a+�:�9?i�|����?�Unknown
�#HostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1      @9      @A      @I      @a{��=��?i8�}�c��?�Unknown
m$HostMul"RMSprop/RMSprop/update_1/mul(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��V�q?i��
���?�Unknown
�%HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��V�q?i�������?�Unknown
X&HostSlice"Slice(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��V�q?iZ^$���?�Unknown
�'HostBiasAddGrad"1gradient_tape/model_4/dense_4/BiasAdd/BiasAddGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��V�q?i)��3��?�Unknown
|(HostReluGrad"&gradient_tape/model_4/dense_4/ReluGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��V�q?i��=����?�Unknown
v)HostCast"$sparse_categorical_crossentropy/Cast(1������	@9������	@A������	@I������	@a��G��?i�,����?�Unknown
�*HostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1������@9������@A������@I������@a���b"?i�ӧ?��?�Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@amsPT�?itWJF���?�Unknown
o,HostMul"RMSprop/RMSprop/update_2/mul_2(1ffffff@9ffffff@Affffff@Iffffff@a�:��?iEI��u��?�Unknown
�-HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      �?A      @I      �?aS�7?i��jV���?�Unknown
w.HostReadVariableOp"RMSprop/Cast/ReadVariableOp(1333333@9333333@A333333@I333333@a7dV��\?is0�;���?�Unknown
�/HostReadVariableOp"%model_4/dense_4/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@a7dV��\?i&�e!��?�Unknown
X0HostEqual"Equal(1ffffff@9ffffff@Affffff@Iffffff@a��:�P\?i������?�Unknown
�1HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a��:�P\?i�����?�Unknown
�2HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a��:�P\?i�%/u|��?�Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1������ @9������ @A������ @I������ @aa/W}3�?iG�����?�Unknown
�4HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1������ @9�������?A������ @I�������?aa/W}3�?i�ˆa��?�Unknown
o5HostSqrt"RMSprop/RMSprop/update_2/Sqrt(1������ @9������ @A������ @I������ @aa/W}3�?i����?�Unknown
s6HostSquare"RMSprop/RMSprop/update_2/Square(1������ @9������ @A������ @I������ @aa/W}3�?i^�f�F��?�Unknown
V7HostSum"Sum_2(1������ @9������ @A������ @I������ @aa/W}3�?i��4!���?�Unknown
^8HostCast"model_4/Cast(1������ @9������ @A������ @I������ @aa/W}3�?i��+��?�Unknown
X9HostCast"Cast_3(1       @9       @A       @I       @a�Q��$E?i�I�����?�Unknown
�:HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1       @9       @A       @I       @a�Q��$E?iB�)���?�Unknown
s;HostSquare"RMSprop/RMSprop/update_3/Square(1       @9       @A       @I       @a�Q��$E?i���r��?�Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?aS���?i�P����?�Unknown
X=HostCast"Cast_2(1�������?9�������?A�������?I�������?aS���?i��?7��?�Unknown
o>HostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?aS���?i��l���?�Unknown
m?HostAddV2"RMSprop/RMSprop/update/add(1333333�?9333333�?A333333�?I333333�?a�����-?i(�#���?�Unknown
o@HostAddV2"RMSprop/RMSprop/update_1/add(1333333�?9333333�?A333333�?I333333�?a�����-?ig-��R��?�Unknown
bAHostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a�����-?i�����?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a��G��?i�k���?�Unknown
�CHostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1�������?9�������?A�������?I�������?a��G��?i�^��?�Unknown
�DHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a��G��?i1�^���?�Unknown
�EHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a��G��?i�Mg���?�Unknown
aFHostIdentity"Identity(1      �?9      �?A      �?I      �?aE����s?i2��q^��?�Unknown�
mGHostMul"RMSprop/RMSprop/update_2/mul(1      �?9      �?A      �?I      �?aE����s?i�bDA���?�Unknown
�HHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�:��?i�[x����?�Unknown
�IHostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�:��?i�T��H��?�Unknown
oJHostAddV2"RMSprop/RMSprop/update_3/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�:��?i�M�R���?�Unknown
qKHostAddV2"RMSprop/RMSprop/update_3/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�:��?i�F����?�Unknown
uLHostRealDiv" RMSprop/RMSprop/update_1/truediv(1�������?9�������?A�������?I�������?a�A�Y��?i���(��?�Unknown
�MHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1�������?9�������?A�������?I�������?a�A�Y��?i�|o��?�Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a7dV��\ ?i�����?�Unknown
VOHostMul"Mul_2(1333333�?9333333�?A333333�?I333333�?a7dV��\ ?ix��a���?�Unknown
�PHostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a7dV��\ ?iҔC�3��?�Unknown
oQHostMul"RMSprop/RMSprop/update_2/mul_1(1333333�?9333333�?A333333�?I333333�?a7dV��\ ?i,jGu��?�Unknown
�RHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a7dV��\ ?i�?�����?�Unknown
`SHostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a7dV��\ ?i��,���?�Unknown
�THostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1�������?9�������?A�������?I�������?a�!B��>irX+4��?�Unknown
�UHostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1�������?9�������?A�������?I�������?a�!B��>i��)p��?�Unknown
qVHostSquare"RMSprop/RMSprop/update/Square(1�������?9�������?A�������?I�������?a�!B��>i��(���?�Unknown
�WHostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1�������?9�������?A�������?I�������?a�!B��>i(#�&���?�Unknown
oXHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1�������?9�������?A�������?I�������?a�!B��>i�f%$��?�Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a�!B��>iL��#`��?�Unknown
�ZHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1      �?9      �?A      �?I      �?a�Q��$E�>i\㭖��?�Unknown
m[HostSqrt"RMSprop/RMSprop/update/Sqrt(1      �?9      �?A      �?I      �?a�Q��$E�>i�-8���?�Unknown
m\HostMul"RMSprop/RMSprop/update/mul_1(1      �?9      �?A      �?I      �?a�Q��$E�>i��v���?�Unknown
o]HostMul"RMSprop/RMSprop/update_3/mul_1(1      �?9      �?A      �?I      �?a�Q��$E�>ixq�L:��?�Unknown
u^HostRealDiv" RMSprop/RMSprop/update_3/truediv(1      �?9      �?A      �?I      �?a�Q��$E�>iC#
�p��?�Unknown
y_HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?aS����>iFC����?�Unknown
�`HostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1�������?9�������?A�������?I�������?aS����>iIc(���?�Unknown
maHostMul"RMSprop/RMSprop/update/mul_2(1�������?9�������?A�������?I�������?aS����>iL�7��?�Unknown
qbHostAddV2"RMSprop/RMSprop/update_1/add_1(1�������?9�������?A�������?I�������?aS����>iO�F/5��?�Unknown
�cHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?aS����>iR�UEf��?�Unknown
�dHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?aS����>iU�d[���?�Unknown
yeHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1�������?9�������?A�������?I�������?a��G���>i�q9����?�Unknown
ofHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1�������?9�������?A�������?I�������?a��G���>i������?�Unknown
sgHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a��G���>i	��@��?�Unknown
ohHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a��G���>iE��E��?�Unknown
miHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a��G���>i����q��?�Unknown
qjHostAddV2"RMSprop/RMSprop/update_2/add_1(1�������?9�������?A�������?I�������?a��G���>i�8`&���?�Unknown
mkHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a��G���>i��4����?�Unknown
ulHostRealDiv" RMSprop/RMSprop/update_2/truediv(1�������?9�������?A�������?I�������?a��G���>i5U	j���?�Unknown
�mHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a��G���>iq�� ��?�Unknown
onHostMul"RMSprop/RMSprop/update_3/mul_2(1�������?9�������?A�������?I�������?a��G���>i�q��K��?�Unknown
moHostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?a��G���>i���Ow��?�Unknown
wpHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a��G���>i%�[���?�Unknown
�qHostReadVariableOp")model_4/MoodOutput/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��G���>ia0����?�Unknown
krHostSub"RMSprop/RMSprop/update/sub(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�:���>i������?�Unknown
�sHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�:���>iId���?�Unknown
ytHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�:���>i��A��?�Unknown
�uHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a7dV��\�>ij|]�a��?�Unknown
ovHostAddV2"RMSprop/RMSprop/update/add_1(1333333�?9333333�?A333333�?I333333�?a7dV��\�>i缎���?�Unknown
swHostRealDiv"RMSprop/RMSprop/update/truediv(1333333�?9333333�?A333333�?I333333�?a7dV��\�>i�QH���?�Unknown
wxHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a7dV��\�>iq�{���?�Unknown
�yHostReadVariableOp"&model_4/dense_4/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a7dV��\�>i'ۺ���?�Unknown
�zHostReadVariableOp"(model_4/MoodOutput/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a�Q��$E�>i     �?�Unknown*�r
uHostFlushSummaryWriter"FlushSummaryWriter(1    ���@9    ���@A    ���@I    ���@a����Wv�?i����Wv�?�Unknown�
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1�����Lb@9�����Lb@Afffffa@Ifffffa@aν���o?i^�Q%��?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(133333�Q@933333�Q@A������E@I������E@a_ǈ|�T?i�^�42��?�Unknown
iHostWriteSummary"WriteSummary(1������9@9������9@A������9@I������9@a�_��VH?i��'�2��?�Unknown�
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1�����;@9�����;@A      9@I      9@a��H�CG?i��h���?�Unknown
�HostBiasAddGrad"4gradient_tape/model_4/MoodOutput/BiasAdd/BiasAddGrad(1�����4@9�����4@A�����4@I�����4@a�G�m�B?i���հ��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������3@9������3@A������3@I������3@ala}�lB?iy�L��?�Unknown
oHostSoftmax"model_4/MoodOutput/Softmax(1fffff�0@9fffff�0@Afffff�0@Ifffff�0@a�]+�-t??inޮ�:��?�Unknown
�	Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������(@9������(@A������(@I������(@a*`�7?i�#��?�Unknown
}
HostMatMul")gradient_tape/model_4/MoodOutput/MatMul_1(1333333(@9333333(@A333333(@I333333(@a���z+�6?i�z�����?�Unknown
nHost_FusedMatMul"model_4/dense_4/Relu(1333333'@9333333'@A333333'@I333333'@aē=��5?i>b�����?�Unknown
xHostMatMul"$gradient_tape/model_4/dense_4/MatMul(1333333$@9333333$@A333333$@I333333$@a��*�?�2?i��'���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      #@9      #@A      #@I      #@a9�"~_�1?i��/��?�Unknown�
dHostDataset"Iterator::Model(1�����=@9�����=@A������"@I������"@a �u�O1?i��$�Y��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������!@9������!@A������!@I������!@a�a0�0?i�
�k��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff%@9ffffff%@Affffff @Iffffff @a&^z��.?i��:ET��?�Unknown
hHostRandomShuffle"RandomShuffle(1333333 @9333333 @A333333 @I333333 @a+�:�&.?i����6��?�Unknown
oHostAddV2"RMSprop/RMSprop/update_2/add(1       @9       @A       @I       @a��v]�-?i��%��?�Unknown
rHostTensorSliceDataset"TensorSliceDataset(1ffffff@9ffffff@Affffff@Iffffff@a�,��F�(?i /:���?�Unknown
{HostMatMul"'gradient_tape/model_4/MoodOutput/MatMul(1ffffff@9ffffff@Affffff@Iffffff@a�,��F�(?isylN%��?�Unknown
�HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1333333@9333333@A333333@I333333@aC��ffs'?il�҄���?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1      @9      @A      @I      @a����U&?i�l4���?�Unknown
\HostSub"RMSprop/sub(1      @9      @A      @I      @a����U&?il��5g��?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1ffffff@9ffffff@Affffff@Iffffff@a�-�[�$?i�TF����?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@aE�۷z�#?iX��b���?�Unknown
ZHostArgMax"ArgMax(1������@9������@A������@I������@a��&�O="?i�$�7��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     @d@9     @d@A������@I������@a��&�O="?i0w�8��?�Unknown
�HostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1������@9������@A������@I������@a`����` ?i}S�>��?�Unknown
tHost_FusedMatMul"model_4/MoodOutput/BiasAdd(1333333@9333333@A333333@I333333@aF�	� ?i��s3>��?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@aZ�Ԉ�D?i�+�W8��?�Unknown
^HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a&^z��?io'P�,��?�Unknown
l HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a\�ڜ�?i��6���?�Unknown
�!HostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1      @9      @A      @I      @a��[�q?i�2�W���?�Unknown
m"HostMul"RMSprop/RMSprop/update_1/mul(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�,��F�?i�����?�Unknown
�#HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�,��F�?iG}0lb��?�Unknown
X$HostSlice"Slice(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�,��F�?i�"f�&��?�Unknown
�%HostBiasAddGrad"1gradient_tape/model_4/dense_4/BiasAdd/BiasAddGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�,��F�?i�Ǜ����?�Unknown
|&HostReluGrad"&gradient_tape/model_4/dense_4/ReluGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�,��F�?i�l�
���?�Unknown
v'HostCast"$sparse_categorical_crossentropy/Cast(1������	@9������	@A������	@I������	@a]�L+��?iX�Z�n��?�Unknown
�(HostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1������@9������@A������@I������@a*`�?i��7A'��?�Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@aē=��?i�P�����?�Unknown
o*HostMul"RMSprop/RMSprop/update_2/mul_2(1ffffff@9ffffff@Affffff@Iffffff@a�-�[�?i���z��?�Unknown
�+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      �?A      @I      �?a���i��?i��g���?�Unknown
w,HostReadVariableOp"RMSprop/Cast/ReadVariableOp(1333333@9333333@A333333@I333333@aŔy��?i^Ҏ����?�Unknown
�-HostReadVariableOp"%model_4/dense_4/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@aŔy��?i+ֵ�-��?�Unknown
X.HostEqual"Equal(1ffffff@9ffffff@Affffff@Iffffff@a�.Wo?i$�0|���?�Unknown
�/HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a�.Wo?iH�w?��?�Unknown
�0HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�.Wo?i&s���?�Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1������ @9������ @A������ @I������ @aZ�Ԉ�D?ii$H�E��?�Unknown
�2HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1������ @9�������?A������ @I�������?aZ�Ԉ�D?i�Gj����?�Unknown
o3HostSqrt"RMSprop/RMSprop/update_2/Sqrt(1������ @9������ @A������ @I������ @aZ�Ԉ�D?ik��?��?�Unknown
s4HostSquare"RMSprop/RMSprop/update_2/Square(1������ @9������ @A������ @I������ @aZ�Ԉ�D?ib������?�Unknown
V5HostSum"Sum_2(1������ @9������ @A������ @I������ @aZ�Ԉ�D?i����9��?�Unknown
^6HostCast"model_4/Cast(1������ @9������ @A������ @I������ @aZ�Ԉ�D?i��߶��?�Unknown
X7HostCast"Cast_3(1       @9       @A       @I       @a��v]�?i��h�-��?�Unknown
�8HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1       @9       @A       @I       @a��v]�?i�����?�Unknown
s9HostSquare"RMSprop/RMSprop/update_3/Square(1       @9       @A       @I       @a��v]�?i�^T8��?�Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a)_�P�
?ia�ql���?�Unknown
X;HostCast"Cast_2(1�������?9�������?A�������?I�������?a)_�P�
?i:䎠���?�Unknown
o<HostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?a)_�P�
?i'��]��?�Unknown
m=HostAddV2"RMSprop/RMSprop/update/add(1333333�?9333333�?A333333�?I333333�?a>�O	?i���?�Unknown
o>HostAddV2"RMSprop/RMSprop/update_1/add(1333333�?9333333�?A333333�?I333333�?a>�O	?i�S(��?�Unknown
b?HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a>�O	?i%�����?�Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a]�L+��?iX������?�Unknown
�AHostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1�������?9�������?A�������?I�������?a]�L+��?i�i�(L��?�Unknown
�BHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a]�L+��?i�Ms���?�Unknown
�CHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a]�L+��?i���
��?�Unknown
aDHostIdentity"Identity(1      �?9      �?A      �?I      �?a����U?iQ&*d��?�Unknown�
mEHostMul"RMSprop/RMSprop/update_2/mul(1      �?9      �?A      �?I      �?a����U?i��Bj���?�Unknown
�FHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�-�[�?i>�����?�Unknown
�GHostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�-�[�?i˷-d��?�Unknown
oHHostAddV2"RMSprop/RMSprop/update_3/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�-�[�?iXφ����?�Unknown
qIHostAddV2"RMSprop/RMSprop/update_3/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�-�[�?i����
��?�Unknown
uJHostRealDiv" RMSprop/RMSprop/update_1/truediv(1�������?9�������?A�������?I�������?a,a.�/[?i���\X��?�Unknown
�KHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1�������?9�������?A�������?I�������?a,a.�/[?iY�rɥ��?�Unknown
vLHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?aŔy��?i?�A���?�Unknown
VMHostMul"Mul_2(1333333�?9333333�?A333333�?I333333�?aŔy��?i%���4��?�Unknown
�NHostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aŔy��?i�1|��?�Unknown
oOHostMul"RMSprop/RMSprop/update_2/mul_1(1333333�?9333333�?A333333�?I333333�?aŔy��?i������?�Unknown
�PHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aŔy��?i�	�!��?�Unknown
`QHostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?aŔy��?i���R��?�Unknown
�RHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1�������?9�������?A�������?I�������?a`����` ?i��N���?�Unknown
�SHostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1�������?9�������?A�������?I�������?a`����` ?i�������?�Unknown
qTHostSquare"RMSprop/RMSprop/update/Square(1�������?9�������?A�������?I�������?a`����` ?i�0$��?�Unknown
�UHostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1�������?9�������?A�������?I�������?a`����` ?i	h��X��?�Unknown
oVHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1�������?9�������?A�������?I�������?a`����` ?i��*���?�Unknown
uWHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a`����` ?i/�R����?�Unknown
�XHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1      �?9      �?A      �?I      �?a��v]��>io�=��?�Unknown
mYHostSqrt"RMSprop/RMSprop/update/Sqrt(1      �?9      �?A      �?I      �?a��v]��>i����R��?�Unknown
mZHostMul"RMSprop/RMSprop/update/mul_1(1      �?9      �?A      �?I      �?a��v]��>iZ���?�Unknown
o[HostMul"RMSprop/RMSprop/update_3/mul_1(1      �?9      �?A      �?I      �?a��v]��>i/�>����?�Unknown
u\HostRealDiv" RMSprop/RMSprop/update_3/truediv(1      �?9      �?A      �?I      �?a��v]��>ios�w��?�Unknown
y]HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?a)_�P��>i�;��?�Unknown
�^HostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1�������?9�������?A�������?I�������?a)_�P��>iI��p��?�Unknown
m_HostMul"RMSprop/RMSprop/update/mul_2(1�������?9�������?A�������?I�������?a)_�P��>i�W%F���?�Unknown
q`HostAddV2"RMSprop/RMSprop/update_1/add_1(1�������?9�������?A�������?I�������?a)_�P��>i#�3����?�Unknown
�aHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a)_�P��>i��Bz��?�Unknown
�bHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?a)_�P��>i�;QG��?�Unknown
ycHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1�������?9�������?A�������?I�������?a]�L+���>i����v��?�Unknown
odHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1�������?9�������?A�������?I�������?a]�L+���>i1�_���?�Unknown
seHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a]�L+���>i�?x���?�Unknown
ofHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a]�L+���>ie�ک��?�Unknown
mgHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a]�L+���>i��<O5��?�Unknown
qhHostAddV2"RMSprop/RMSprop/update_2/add_1(1�������?9�������?A�������?I�������?a]�L+���>i�C��d��?�Unknown
miHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a]�L+���>i3�����?�Unknown
ujHostRealDiv" RMSprop/RMSprop/update_2/truediv(1�������?9�������?A�������?I�������?a]�L+���>i��c?���?�Unknown
�kHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a]�L+���>igG�����?�Unknown
olHostMul"RMSprop/RMSprop/update_3/mul_2(1�������?9�������?A�������?I�������?a]�L+���>i�(�#��?�Unknown
mmHostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?a]�L+���>i��/S��?�Unknown
wnHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a]�L+���>i5K�Ԃ��?�Unknown
�oHostReadVariableOp")model_4/MoodOutput/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a]�L+���>iϡOz���?�Unknown
kpHostSub"RMSprop/RMSprop/update/sub(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�-�[��>i��+���?�Unknown
�qHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�-�[��>i[�����?�Unknown
yrHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�-�[��>i!�q�/��?�Unknown
�sHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aŔy���>i�{HS��?�Unknown
otHostAddV2"RMSprop/RMSprop/update/add_1(1333333�?9333333�?A333333�?I333333�?aŔy���>iG�w��?�Unknown
suHostRealDiv"RMSprop/RMSprop/update/truediv(1333333�?9333333�?A333333�?I333333�?aŔy���>i������?�Unknown
wvHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aŔy���>i�Ș|���?�Unknown
�wHostReadVariableOp"&model_4/dense_4/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aŔy���>i���8���?�Unknown
�xHostReadVariableOp"(model_4/MoodOutput/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a��v]��>i      �?�Unknown2GPU