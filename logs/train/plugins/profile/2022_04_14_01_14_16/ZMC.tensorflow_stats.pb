"�t
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1����l��@9����l��@A����l��@I����l��@a|��W���?i|��W���?�Unknown�
BHostIDLE"IDLE1fffff$�@Afffff$�@a�/�{3Q�?if��6@8�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff�e@9fffff�e@A�����id@I�����id@a�E�ewr?i΃�/]�?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1�����<`@9�����<`@A33333Y@I33333Y@a4�XI3�f?ih�D5�s�?�Unknown
iHostWriteSummary"WriteSummary(1333333A@9333333A@A333333A@I333333A@a�f~��O?i���{�?�Unknown�
pHost_FusedMatMul"model_11/dense_11/Relu(1ffffff=@9ffffff=@Affffff=@Iffffff=@a]�PטJ?i���L��?�Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_1(1������;@9������;@A������;@I������;@a=� '��H?i����?�Unknown
�	HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1������=@9������=@A�����;@I�����;@a8-�H?iƯZ(���?�Unknown
~
HostMatMul"*gradient_tape/model_11/MoodOutput/MatMul_1(1     �:@9     �:@A     �:@I     �:@a�M�8�G?iY�zv���?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      6@9      6@A      6@I      6@ah���C?i���9���?�Unknown
pHostSoftmax"model_11/MoodOutput/Softmax(1fffff�5@9fffff�5@Afffff�5@Ifffff�5@a-%ZN��C?i�2���?�Unknown
bHostDivNoNan"div_no_nan_1(133333�4@933333�4@A33333�4@I33333�4@an�X���B?iLp��E��?�Unknown
|HostMatMul"(gradient_tape/model_11/MoodOutput/MatMul(1      4@9      4@A      4@I      4@a�5m[�B?i�K�˧�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(133333�3@933333�3@A33333�3@I33333�3@a#�,�b�A?i��A@��?�Unknown
uHost_FusedMatMul"model_11/MoodOutput/BiasAdd(1fffff�1@9fffff�1@Afffff�1@Ifffff�1@a��Մ1@?i}�עL��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1�����1@9�����1@A      ,@I      ,@a(�L�T9?i a6w��?�Unknown
zHostMatMul"&gradient_tape/model_11/dense_11/MatMul(1������+@9������+@A������+@I������+@a=� '��8?i��5���?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      $@9      $@A      $@I      $@a�5m[�2?i�l11ٸ�?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1������#@9������#@A������#@I������#@a^��ȋ�1?i��b��?�Unknown
~HostReluGrad"(gradient_tape/model_11/dense_11/ReluGrad(1������#@9������#@A������#@I������#@a��5:�1?i�7��M��?�Unknown
^HostGatherV2"GatherV2(1ffffff#@9ffffff#@Affffff#@Iffffff#@as~��1?i��g��?�Unknown
hHostRandomShuffle"RandomShuffle(1333333#@9333333#@A333333#@I333333#@a���^1?ih��9���?�Unknown
mHostMul"RMSprop/RMSprop/update_1/mul(1������!@9������!@A������!@I������!@a�����/?i��6����?�Unknown
dHostDataset"Iterator::Model(1333333>@9333333>@Affffff @Iffffff @a&��B�-?i��W���?�Unknown
rHostTensorSliceDataset"TensorSliceDataset(1333333 @9333333 @A333333 @I333333 @a<�&�O-?i;�FyX��?�Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a���.o�+?i*r9���?�Unknown
\HostSub"RMSprop/sub(1      @9      @A      @I      @a��#	�#+?ig�����?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@aӫx�(�*?i�;�/o��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�h@933333�h@A333333@I333333@a�ͽ�j*?i�����?�Unknown
�HostBiasAddGrad"3gradient_tape/model_11/dense_11/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a(�L�T)?i��!���?�Unknown
m HostMul"RMSprop/RMSprop/update_3/mul(1������@9������@A������@I������@a=� '��(?i�W;�:��?�Unknown
�!HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1������@9������@A������@I������@ah��۲>(?iFi����?�Unknown
e"Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a�E&�&?i'f�N+��?�Unknown�
�#HostBiasAddGrad"5gradient_tape/model_11/MoodOutput/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a�����&?i��|���?�Unknown
_$HostCast"model_11/Cast(1������@9������@A������@I������@a�q��Y%?i��g���?�Unknown
Z%HostArgMax"ArgMax(1      @9      @A      @I      @ah���#?if9� ��?�Unknown
`&HostGatherV2"
GatherV2_1(1      @9      @A      @I      @ah���#?iZ�
�^��?�Unknown
v'HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @ah���#?i�^�h���?�Unknown
m(HostSqrt"RMSprop/RMSprop/update/Sqrt(1ffffff@9ffffff@Affffff@Iffffff@a�Z��t"?i/p����?�Unknown
�)HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1������@9������@A������@I������@a�k��!?i�#����?�Unknown
m*HostMul"RMSprop/RMSprop/update_2/mul(1ffffff@9ffffff@Affffff@Iffffff@a(���P� ?i�b/%���?�Unknown
q+HostSquare"RMSprop/RMSprop/update/Square(1      @9      @A      @I      @aR�{���?iю����?�Unknown
y,HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1333333@9333333@A333333@I333333@a}?%z�9?i�_�����?�Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a���.o�?is�2����?�Unknown
�.HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1������@9������@A������@I������@aӫx�(�?i8�y�Z��?�Unknown
X/HostSlice"Slice(1������@9������@A������@I������@aӫx�(�?i�� 1��?�Unknown
X0HostEqual"Equal(1      @9      @A      @I      @a(�L�T?i^t�����?�Unknown
y1HostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a}���?iZ%!����?�Unknown
m2HostMul"RMSprop/RMSprop/update/mul_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@a}���?iV֞�y��?�Unknown
�3HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a}���?iR��8��?�Unknown
�4HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a��r�o?i�5S���?�Unknown
�5HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a���<�?i�#���?�Unknown
o6HostMul"RMSprop/RMSprop/update_2/mul_2(1333333@9333333@A333333@I333333@a(]ƈ��?ij��A��?�Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff@9ffffff@Affffff@Iffffff@aSp=�C?i�UR
���?�Unknown
X8HostCast"Cast_3(1������@9������@A������@I������@a�æ#�?i��o�z��?�Unknown
�9HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������@9������@A������@I������@a�æ#�?i�����?�Unknown
�:HostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1������@9������@A������@I������@a�æ#�?i�������?�Unknown
t;HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a�5m[�?iAӔd8��?�Unknown
�<HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1      @9      �?A      @I      �?a�5m[�?i��#���?�Unknown
�=HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1      @9      @A      @I      @a�5m[�?i�j�Y��?�Unknown
�>HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1      @9      @A      @I      @a�5m[�?ieU����?�Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_3(1333333@9333333@A333333@I333333@a���^?i6��u��?�Unknown
k@HostSub"RMSprop/RMSprop/update/sub(1333333@9333333@A333333@I333333@a���^?i�fƊ ��?�Unknown
�AHostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1333333@9333333@A333333@I333333@a���^?i��~���?�Unknown
�BHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1333333@9333333@A333333@I333333@a���^?i[h7t��?�Unknown
�CHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a(���P�?i`������?�Unknown
mDHostAddV2"RMSprop/RMSprop/update/add(1ffffff@9ffffff@Affffff@Iffffff@a(���P�?ie�C� ��?�Unknown
sEHostSquare"RMSprop/RMSprop/update_2/Square(1������@9������@A������@I������@a�����?i��)���?�Unknown
�FHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1������@9������@A������@I������@a�����?iK���?�Unknown
XGHostCast"Cast_2(1������ @9������ @A������ @I������ @a�(\�e?i�� ���?�Unknown
oHHostMul"RMSprop/RMSprop/update_3/mul_2(1������ @9������ @A������ @I������ @a�(\�e?iK,.���?�Unknown
�IHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1       @9       @A       @I       @aR�{���?i9B����?�Unknown
oJHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1       @9       @A       @I       @aR�{���?i'XN���?�Unknown
sKHostSquare"RMSprop/RMSprop/update_3/Square(1       @9       @A       @I       @aR�{���?in�n��?�Unknown
�LHostReadVariableOp"'model_11/dense_11/MatMul/ReadVariableOp(1       @9       @A       @I       @aR�{���?i������?�Unknown
�MHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���.o�?i??��O��?�Unknown
�NHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���.o�?i{�c���?�Unknown
oOHostSqrt"RMSprop/RMSprop/update_2/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���.o�?i�� �+��?�Unknown
uPHostRealDiv" RMSprop/RMSprop/update_3/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���.o�?i�p����?�Unknown
�QHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���.o�?i/,����?�Unknown
oRHostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?a�a"��
?i��$&p��?�Unknown
�SHostReadVariableOp")model_11/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�a"��
?iC��]���?�Unknown
`THostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?aR�uV�?i��:��?�Unknown
sUHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a�:�j�(?i?�,n���?�Unknown
mVHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a�:�j�(?idIR���?�Unknown
�WHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      �?9      �?A      �?I      �?a���<�?iיE�J��?�Unknown
�XHostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1      �?9      �?A      �?I      �?a���<�?iJ�8á��?�Unknown
mYHostMul"RMSprop/RMSprop/update/mul_1(1      �?9      �?A      �?I      �?a���<�?i�:,����?�Unknown
oZHostAddV2"RMSprop/RMSprop/update_2/add(1      �?9      �?A      �?I      �?a���<�?i0�uO��?�Unknown
w[HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?a���<�?i��N���?�Unknown
w\HostReadVariableOp"RMSprop/Cast/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSp=�C?ic��\���?�Unknown
o]HostAddV2"RMSprop/RMSprop/update_3/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSp=�C?i#ǔkH��?�Unknown
s^HostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?a�æ#�?i1b#����?�Unknown
�_HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a�æ#�?i?������?�Unknown
o`HostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?a�æ#�?iM�@9*��?�Unknown
oaHostMul"RMSprop/RMSprop/update_3/mul_1(1�������?9�������?A�������?I�������?a�æ#�?i[3�}u��?�Unknown
mbHostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?a�æ#�?ii�]����?�Unknown
ocHostAddV2"RMSprop/RMSprop/update/add_1(1333333�?9333333�?A333333�?I333333�?a���^?i��<��?�Unknown
qdHostAddV2"RMSprop/RMSprop/update_2/add_1(1333333�?9333333�?A333333�?I333333�?a���^?i!O�K��?�Unknown
ueHostRealDiv" RMSprop/RMSprop/update_2/truediv(1333333�?9333333�?A333333�?I333333�?a���^?i}�r1���?�Unknown
qfHostAddV2"RMSprop/RMSprop/update_3/add_1(1333333�?9333333�?A333333�?I333333�?a���^?i��Ϋ���?�Unknown
TgHostMul"Mul(1�������?9�������?A�������?I�������?a������>i���[��?�Unknown
ohHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1�������?9�������?A�������?I�������?a������>i+�"V��?�Unknown
qiHostAddV2"RMSprop/RMSprop/update_1/add_1(1�������?9�������?A�������?I�������?a������>iԀL����?�Unknown
ojHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a������>i}fvl���?�Unknown
�kHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?a������>i&L���?�Unknown
�lHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?aR�{����>iחO��?�Unknown
omHostAddV2"RMSprop/RMSprop/update_1/add(1      �?9      �?A      �?I      �?aR�{����>ib����?�Unknown
mnHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a�a"���>iY�T���?�Unknown
uoHostRealDiv" RMSprop/RMSprop/update_1/truediv(1�������?9�������?A�������?I�������?a�a"���>i�� ���?�Unknown
�pHostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1�������?9�������?A�������?I�������?a�a"���>i���;%��?�Unknown
�qHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a�a"���>i(#�WY��?�Unknown
wrHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�a"���>imSis���?�Unknown
ysHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�a"���>i��.����?�Unknown
�tHostReadVariableOp"(model_11/dense_11/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�a"���>i������?�Unknown
auHostIdentity"Identity(1�������?9�������?A�������?I�������?a�:�j�(�>i����#��?�Unknown�
�vHostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1�������?9�������?A�������?I�������?a�:�j�(�>i_NR��?�Unknown
�wHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a�:�j�(�>i�4�����?�Unknown
�xHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a�:�j�(�>i?
?���?�Unknown
uyHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSp=�C�>i��x���?�Unknown
�zHostReadVariableOp"*model_11/MoodOutput/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSp=�C�>i�������?�Unknown*�s
uHostFlushSummaryWriter"FlushSummaryWriter(1����l��@9����l��@A����l��@I����l��@a�����?i�����?�Unknown�
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff�e@9fffff�e@A�����id@I�����id@a���[u?i�e���C�?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1�����<`@9�����<`@A33333Y@I33333Y@aa{OD�;j?i��x�]�?�Unknown
iHostWriteSummary"WriteSummary(1333333A@9333333A@A333333A@I333333A@a�7�g��Q?i$����f�?�Unknown�
pHost_FusedMatMul"model_11/dense_11/Relu(1ffffff=@9ffffff=@Affffff=@Iffffff=@a�U7#�N?i�mwS�n�?�Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_1(1������;@9������;@A������;@I������;@a��)�L?i{́V�u�?�Unknown
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1������=@9������=@A�����;@I�����;@a3~�3"ZL?i����|�?�Unknown
~HostMatMul"*gradient_tape/model_11/MoodOutput/MatMul_1(1     �:@9     �:@A     �:@I     �:@afj��o�K?i6A;ރ�?�Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1      6@9      6@A      6@I      6@a��V76G?i��H���?�Unknown
p
HostSoftmax"model_11/MoodOutput/Softmax(1fffff�5@9fffff�5@Afffff�5@Ifffff�5@a���m�F?i���Y��?�Unknown
bHostDivNoNan"div_no_nan_1(133333�4@933333�4@A33333�4@I33333�4@a�	�E?i˟O�Ô�?�Unknown
|HostMatMul"(gradient_tape/model_11/MoodOutput/MatMul(1      4@9      4@A      4@I      4@a��`��D?ip������?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(133333�3@933333�3@A33333�3@I33333�3@a��35�D?i��4�%��?�Unknown
uHost_FusedMatMul"model_11/MoodOutput/BiasAdd(1fffff�1@9fffff�1@Afffff�1@Ifffff�1@a�N&�B?i�Q�^ԣ�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1�����1@9�����1@A      ,@I      ,@a�˺-K=?i�u�}��?�Unknown
zHostMatMul"&gradient_tape/model_11/dense_11/MatMul(1������+@9������+@A������+@I������+@a��)�<?i�����?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      $@9      $@A      $@I      $@a��`��4?i^��W���?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1������#@9������#@A������#@I������#@aS�4���4?i��y7N��?�Unknown
~HostReluGrad"(gradient_tape/model_11/dense_11/ReluGrad(1������#@9������#@A������#@I������#@a����l�4?i��e޲�?�Unknown
^HostGatherV2"GatherV2(1ffffff#@9ffffff#@Affffff#@Iffffff#@a ���K4?i�˔�g��?�Unknown
hHostRandomShuffle"RandomShuffle(1333333#@9333333#@A333333#@I333333#@a�yB>K4?i �����?�Unknown
mHostMul"RMSprop/RMSprop/update_1/mul(1������!@9������!@A������!@I������!@a�D��i2?ii���7��?�Unknown
dHostDataset"Iterator::Model(1333333>@9333333>@Affffff @Iffffff @a#.E`(1?i-\��\��?�Unknown
rHostTensorSliceDataset"TensorSliceDataset(1333333 @9333333 @A333333 @I333333 @a��|��0?i0�H{��?�Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a�"��/?io(x��?�Unknown
\HostSub"RMSprop/sub(1      @9      @A      @I      @a��G��b/?i�*_Un��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a������.?i�*��]��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�h@933333�h@A333333@I333333@a}ïn��.?i�ƙF��?�Unknown
�HostBiasAddGrad"3gradient_tape/model_11/dense_11/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a�˺-K-?i`¡L��?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1������@9������@A������@I������@a��)�,?iYZdM���?�Unknown
�HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1������@9������@A������@I������@aLt��	,?i������?�Unknown
e Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a�?��B]*?iD� �O��?�Unknown�
�!HostBiasAddGrad"5gradient_tape/model_11/MoodOutput/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a%���)?i6�.���?�Unknown
_"HostCast"model_11/Cast(1������@9������@A������@I������@a�
�|��(?i���9s��?�Unknown
Z#HostArgMax"ArgMax(1      @9      @A      @I      @a��V76'?iF}���?�Unknown
`$HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a��V76'?i����S��?�Unknown
v%HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a��V76'?i�����?�Unknown
m&HostSqrt"RMSprop/RMSprop/update/Sqrt(1ffffff@9ffffff@Affffff@Iffffff@a�&�W%?iX!��?�Unknown
�'HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1������@9������@A������@I������@aTl��)�#?i��1T��?�Unknown
m(HostMul"RMSprop/RMSprop/update_2/mul(1ffffff@9ffffff@Affffff@Iffffff@a!_�@#?ie�(2���?�Unknown
q)HostSquare"RMSprop/RMSprop/update/Square(1      @9      @A      @I      @a��>� ?i�����?�Unknown
y*HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1333333@9333333@A333333@I333333@a��"R ?i��'���?�Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a�"��?i�'�����?�Unknown
�,HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1������@9������@A������@I������@a������?id'?U���?�Unknown
X-HostSlice"Slice(1������@9������@A������@I������@a������?iC'����?�Unknown
X.HostEqual"Equal(1      @9      @A      @I      @a�˺-K?i��Llq��?�Unknown
y/HostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@ag�u��?i{��aN��?�Unknown
m0HostMul"RMSprop/RMSprop/update/mul_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@ag�u��?iVW�V+��?�Unknown
�1HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@ag�u��?i1 L��?�Unknown
�2HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?aN2k0!�?i��	����?�Unknown
�3HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a���?i$������?�Unknown
o4HostMul"RMSprop/RMSprop/update_2/mul_2(1333333@9333333@A333333@I333333@a��:�E?i�O��b��?�Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff@9ffffff@Affffff@Iffffff@a��Wo?i��c��?�Unknown
X6HostCast"Cast_3(1������@9������@A������@I������@aR�r���?i��z���?�Unknown
�7HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������@9������@A������@I������@aR�r���?i=̧�z��?�Unknown
�8HostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1������@9������@A������@I������@aR�r���?i��3�(��?�Unknown
t9HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a��`��?i�����?�Unknown
�:HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1      @9      �?A      @I      �?a��`��?i|�pw��?�Unknown
�;HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1      @9      @A      @I      @a��`��?iQ�����?�Unknown
�<HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1      @9      @A      @I      @a��`��?i& 9���?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1333333@9333333@A333333@I333333@a�yB>K?i:�Y�f��?�Unknown
k>HostSub"RMSprop/RMSprop/update/sub(1333333@9333333@A333333@I333333@a�yB>K?iN糝��?�Unknown
�?HostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1333333@9333333@A333333@I333333@a�yB>K?ib�P���?�Unknown
�@HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1333333@9333333@A333333@I333333@a�yB>K?iv�gI��?�Unknown
�AHostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a!_�@?iɨ����?�Unknown
mBHostAddV2"RMSprop/RMSprop/update/add(1ffffff@9ffffff@Affffff@Iffffff@a!_�@?i��}��?�Unknown
sCHostSquare"RMSprop/RMSprop/update_2/Square(1������@9������@A������@I������@a�D��i?i�NQ��?�Unknown
�DHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1������@9������@A������@I������@a�D��i?i@9����?�Unknown
XEHostCast"Cast_2(1������ @9������ @A������ @I������ @aV*zց�?i�G;0��?�Unknown
oFHostMul"RMSprop/RMSprop/update_3/mul_2(1������ @9������ @A������ @I������ @aV*zց�?i�~V׼��?�Unknown
�GHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1       @9       @A       @I       @a��>�?i�L�B��?�Unknown
oHHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1       @9       @A       @I       @a��>�?i�A����?�Unknown
sIHostSquare"RMSprop/RMSprop/update_3/Square(1       @9       @A       @I       @a��>�?i\7�N��?�Unknown
�JHostReadVariableOp"'model_11/dense_11/MatMul/ReadVariableOp(1       @9       @A       @I       @a��>�?i"�,���?�Unknown
�KHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�"��?ir�	�S��?�Unknown
�LHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�"��?i������?�Unknown
oMHostSqrt"RMSprop/RMSprop/update_2/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�"��?i��&R��?�Unknown
uNHostRealDiv" RMSprop/RMSprop/update_3/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�"��?ib$�^���?�Unknown
�OHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�"��?i��{�P��?�Unknown
oPHostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?aK�c�p!?iA$?���?�Unknown
�QHostReadVariableOp")model_11/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aK�c�p!?iЙ�A��?�Unknown
`RHostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a�3��t?i���u���?�Unknown
sSHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a�LSd�
?i�F>���?�Unknown
mTHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a�LSd�
?i��ϸ���?�Unknown
�UHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      �?9      �?A      �?I      �?a���	?i�G(���?�Unknown
�VHostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1      �?9      �?A      �?I      �?a���	?iP��R��?�Unknown
mWHostMul"RMSprop/RMSprop/update/mul_1(1      �?9      �?A      �?I      �?a���	?i�88���?�Unknown
oXHostAddV2"RMSprop/RMSprop/update_2/add(1      �?9      �?A      �?I      �?a���	?i�o�v��?�Unknown
wYHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?a���	?i4�(���?�Unknown
wZHostReadVariableOp"RMSprop/Cast/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��Wo?i�ɇ����?�Unknown
o[HostAddV2"RMSprop/RMSprop/update_3/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��Wo?iL��`;��?�Unknown
s\HostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?aR�r���?i�,l���?�Unknown
�]HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1�������?9�������?A�������?I�������?aR�r���?i�sw���?�Unknown
o^HostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?aR�r���?i���@��?�Unknown
o_HostMul"RMSprop/RMSprop/update_3/mul_1(1�������?9�������?A�������?I�������?aR�r���?ix#�����?�Unknown
m`HostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?aR�r���?iC1E����?�Unknown
oaHostAddV2"RMSprop/RMSprop/update/add_1(1333333�?9333333�?A333333�?I333333�?a�yB>K?iM*r�>��?�Unknown
qbHostAddV2"RMSprop/RMSprop/update_2/add_1(1333333�?9333333�?A333333�?I333333�?a�yB>K?iW#�K���?�Unknown
ucHostRealDiv" RMSprop/RMSprop/update_2/truediv(1333333�?9333333�?A333333�?I333333�?a�yB>K?ia̤���?�Unknown
qdHostAddV2"RMSprop/RMSprop/update_3/add_1(1333333�?9333333�?A333333�?I333333�?a�yB>K?ik��/��?�Unknown
TeHostMul"Mul(1�������?9�������?A�������?I�������?a�D��i?i���y��?�Unknown
ofHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1�������?9�������?A�������?I�������?a�D��i?i�� L���?�Unknown
qgHostAddV2"RMSprop/RMSprop/update_1/add_1(1�������?9�������?A�������?I�������?a�D��i?iF�4���?�Unknown
ohHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a�D��i?i��H�V��?�Unknown
�iHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1�������?9�������?A�������?I�������?a�D��i?i؊\A���?�Unknown
�jHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?a��>� ?i`ZW6���?�Unknown
okHostAddV2"RMSprop/RMSprop/update_1/add(1      �?9      �?A      �?I      �?a��>� ?i�)R+&��?�Unknown
mlHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?aK�c�p!�>i��3nb��?�Unknown
umHostRealDiv" RMSprop/RMSprop/update_1/truediv(1�������?9�������?A�������?I�������?aK�c�p!�>iv�����?�Unknown
�nHostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1�������?9�������?A�������?I�������?aK�c�p!�>i=Z�����?�Unknown
�oHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?aK�c�p!�>i�6��?�Unknown
wpHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?aK�c�p!�>i�ϺyS��?�Unknown
yqHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?aK�c�p!�>i�������?�Unknown
�rHostReadVariableOp"(model_11/dense_11/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?aK�c�p!�>iYE~����?�Unknown
asHostIdentity"Identity(1�������?9�������?A�������?I�������?a�LSd��>i`�F���?�Unknown�
�tHostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1�������?9�������?A�������?I�������?a�LSd��>ig�!7��?�Unknown
�uHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a�LSd��>in7رl��?�Unknown
�vHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a�LSd��>iuݠB���?�Unknown
uwHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��Wo�>i�nP!���?�Unknown
�xHostReadVariableOp"*model_11/MoodOutput/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��Wo�>i      �?�Unknown2GPU