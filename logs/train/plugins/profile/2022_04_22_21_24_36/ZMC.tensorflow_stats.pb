"�k
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1    ��@9    ��@A    ��@I    ��@a������?i������?�Unknown�
BHostIDLE"IDLE1�����Ӯ@A�����Ӯ@a;��)�?i2k�b���?�Unknown
iHostWriteSummary"WriteSummary(1ffffff6@9ffffff6@Affffff6@Iffffff6@a��TPhvD?ih��Û�?�Unknown�
pHostSoftmax"model_39/MoodOutput/Softmax(1������5@9������5@A������5@I������5@abvMR�C?iޙѲ��?�Unknown
pHost_FusedMatMul"model_39/dense_39/Relu(13333334@93333334@A3333334@I3333334@a�q��sB?iF���O��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333�6@933333�6@A      3@I      3@a�d#�J[A?i=����?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(133333�1@933333�1@A33333�1@I33333�1@a��y�F+@?i���p���?�Unknown
~	HostMatMul"*gradient_tape/model_39/MoodOutput/MatMul_1(1      0@9      0@A      0@I      0@asX�rp;=?i�9�X��?�Unknown
�
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      .@9      .@A      .@I      .@a�2�k�g;?i��8�Ŵ�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff+@9ffffff+@Affffff+@Iffffff+@aU5C�9?i�t���?�Unknown
|HostMatMul"(gradient_tape/model_39/MoodOutput/MatMul(1������(@9������(@A������(@I������(@ae�8��x6?i#�鵺�?�Unknown
dHostDataset"Iterator::Model(1      =@9      =@A������&@I������&@a�t�-�4?i�ʎJ��?�Unknown
\HostSub"RMSprop/sub(1333333&@9333333&@A333333&@I333333&@a):�ϢG4?iY$�ӿ�?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1ffffff$@9ffffff$@Affffff$@Iffffff$@a��(I��2?io6M�'��?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1ffffff"@9ffffff"@Affffff"@Iffffff"@au��A��0?iv��A��?�Unknown
zHostMatMul"&gradient_tape/model_39/dense_39/MatMul(1      "@9      "@A      "@I      "@a�Q�@oq0?i��}�O��?�Unknown
`HostGatherV2"
GatherV2_1(1������!@9������!@A������!@I������!@af�տ�B0?ib��;X��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff @9ffffff @Affffff @Iffffff @a�͠u��-?io��7��?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������(@9������(@A333333@I333333@a	��oZ�,?i��é���?�Unknown
�HostBiasAddGrad"5gradient_tape/model_39/MoodOutput/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a΂HgO*?i5OJ����?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @adjd�)?i֕p�=��?�Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@a���a��(?i��6j���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a�7�X��&?i�=��5��?�Unknown�
�HostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a
��WJ&?i��ц���?�Unknown
_HostCast"model_39/Cast(1ffffff@9ffffff@Affffff@Iffffff@a
��WJ&?i�-�(���?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�C@933333�C@A������@I������@a�<JF��!?iV�{���?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1������@9������@A������@I������@a�<JF��!?i��/<��?�Unknown
`HostDivNoNan"
div_no_nan(1ffffff@9ffffff@Affffff@Iffffff@au��A�� ?i��I��?�Unknown
~HostReluGrad"(gradient_tape/model_39/dense_39/ReluGrad(1ffffff@9ffffff@Affffff@Iffffff@au��A�� ?i�6x�U��?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a�Q�@oq ?ia?l]��?�Unknown
Z HostArgMax"ArgMax(1������@9������@A������@I������@a?� ?iB1�P^��?�Unknown
m!HostMul"RMSprop/RMSprop/update_2/mul(1������@9������@A������@I������@a?� ?i##�_��?�Unknown
�"HostBiasAddGrad"3gradient_tape/model_39/dense_39/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a?� ?i8�`��?�Unknown
v#HostCast"$sparse_categorical_crossentropy/Cast(1������@9������@A������@I������@a?� ?i�|b��?�Unknown
X$HostCast"Cast_3(1������@9������@A������@I������@aFCx��?i��_�W��?�Unknown
m%HostMul"RMSprop/RMSprop/update_1/mul(1333333@9333333@A333333@I333333@a	��oZ�?i�I3�;��?�Unknown
s&HostSquare"RMSprop/RMSprop/update_3/Square(1333333@9333333@A333333@I333333@a	��oZ�?i����?�Unknown
u'Host_FusedMatMul"model_39/MoodOutput/BiasAdd(1333333@9333333@A333333@I333333@a	��oZ�?i<Hڠ��?�Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff@9ffffff@Affffff@Iffffff@a�mmD�?ig������?�Unknown
�)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a7�&j.
?i�q���?�Unknown
�*HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1������@9������@A������@I������@a΂HgO?i�;4����?�Unknown
�+HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1������@9������@A������@I������@a΂HgO?i'v�_��?�Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�"�^�?i�k����?�Unknown
o-HostMul"RMSprop/RMSprop/update_3/mul_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�"�^�?i�`]����?�Unknown
�.HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1������	@9������	@A������	@I������	@a)��[�b?in?`���?�Unknown
y/HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1      @9      @A      @I      @aV�V��?i��fK��?�Unknown
X0HostCast"Cast_2(1333333@9333333@A333333@I333333@a�L3S~1?i�������?�Unknown
�1HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1333333@9333333@A333333@I333333@a�L3S~1?i0#�}���?�Unknown
o2HostAddV2"RMSprop/RMSprop/update_2/add(1333333@9333333@A333333@I333333@a�L3S~1?iʼ�	H��?�Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff@9ffffff@Affffff@Iffffff@a��TPhv?iq?����?�Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_1(1������@9������@A������@I������@abvMR�?i$������?�Unknown
�5HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1������@9������@A������@I������@abvMR�?i�Br'��?�Unknown
�6HostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1333333@9333333@A333333@I333333@a��D�?i�=�³��?�Unknown
o7HostMul"RMSprop/RMSprop/update_2/mul_2(1333333@9333333@A333333@I333333@a��D�?i�dF@��?�Unknown
o8HostSqrt"RMSprop/RMSprop/update_3/Sqrt(1333333@9333333@A333333@I333333@a��D�?i_��c���?�Unknown
o9HostSqrt"RMSprop/RMSprop/update_2/Sqrt(1ffffff@9ffffff@Affffff@Iffffff@au��A��?iC���R��?�Unknown
�:HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@au��A��?i'�lS���?�Unknown
V;HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@au��A��?i�>�_��?�Unknown
�<HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1������@9������@A������@I������@a?�?i��`j���?�Unknown
X=HostEqual"Equal(1������ @9������ @A������ @I������ @aFCx��?i���0[��?�Unknown
w>HostReadVariableOp"RMSprop/Cast/ReadVariableOp(1������ @9������ @A������ @I������ @aFCx��?i�wD����?�Unknown
�?HostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1������ @9������ @A������ @I������ @aFCx��?i�Y��P��?�Unknown
s@HostSquare"RMSprop/RMSprop/update_2/Square(1������ @9������ @A������ @I������ @aFCx��?i�;(����?�Unknown
uAHostReadVariableOp"div_no_nan/ReadVariableOp(1������ @9������ @A������ @I������ @aFCx��?i��JF��?�Unknown
qBHostSquare"RMSprop/RMSprop/update/Square(1       @9       @A       @I       @asX�rp;?i��[8���?�Unknown
�CHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @asX�rp;?i��&0��?�Unknown
TDHostMul"Mul(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�mmD�?ih/;���?�Unknown
yEHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�mmD�?i+AP��?�Unknown
�FHostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�mmD�?iA�Re}��?�Unknown
�GHostReadVariableOp"'model_39/dense_39/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�mmD�?iW�dz���?�Unknown
oHHostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?a΂HgO
?iy!ƶU��?�Unknown
�IHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1�������?9�������?A�������?I�������?a΂HgO
?i��'���?�Unknown
qJHostAddV2"RMSprop/RMSprop/update_3/add_1(1�������?9�������?A�������?I�������?a΂HgO
?i�[�/(��?�Unknown
uKHostRealDiv" RMSprop/RMSprop/update_3/truediv(1�������?9�������?A�������?I�������?a΂HgO
?i���k���?�Unknown
�LHostReadVariableOp")model_39/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a΂HgO
?i�L����?�Unknown
�MHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1333333�?9333333�?A333333�?I333333�?a���a��?i/�^��?�Unknown
�NHostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1�������?9�������?A�������?I�������?a)��[�b?ij������?�Unknown
�OHostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1      �?9      �?A      �?I      �?aV�V��?i��PI��?�Unknown
mPHostAddV2"RMSprop/RMSprop/update/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��TPhv?i%�"e��?�Unknown
mQHostMul"RMSprop/RMSprop/update/mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��TPhv?iWf�����?�Unknown
�RHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��TPhv?i��4���?�Unknown
�SHostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a��J< ?i
�%�T��?�Unknown
kTHostSub"RMSprop/RMSprop/update/sub(1�������?9�������?A�������?I�������?a��J< ?ij�ؠ��?�Unknown
oUHostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?a��J< ?i�&����?�Unknown
wVHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a��J< ?i*Q��8��?�Unknown
�WHostReadVariableOp"(model_39/dense_39/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��J< ?i�{�ڄ��?�Unknown
�XHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1333333�?9333333�?A333333�?I333333�?a��D�?i��+���?�Unknown
oYHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1333333�?9333333�?A333333�?I333333�?a��D�?ib�l+��?�Unknown
�ZHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��D�?iε�SW��?�Unknown
o[HostAddV2"RMSprop/RMSprop/update_3/add(1333333�?9333333�?A333333�?I333333�?a��D�?i:��{���?�Unknown
o\HostMul"RMSprop/RMSprop/update_3/mul_1(1333333�?9333333�?A333333�?I333333�?a��D�?i��/����?�Unknown
b]HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a��D�?i�p�)��?�Unknown
�^HostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?asX�rp;�>i��QCd��?�Unknown
�_HostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?asX�rp;�>i�2����?�Unknown
m`HostSqrt"RMSprop/RMSprop/update/Sqrt(1�������?9�������?A�������?I�������?a΂HgO�>i��cX���?�Unknown
saHostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?a΂HgO�>i>X����?�Unknown
sbHostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a΂HgO�>i�&Ŕ<��?�Unknown
ocHostAddV2"RMSprop/RMSprop/update_1/add(1�������?9�������?A�������?I�������?a΂HgO�>i`��2q��?�Unknown
odHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a΂HgO�>i��&ѥ��?�Unknown
qeHostAddV2"RMSprop/RMSprop/update_2/add_1(1�������?9�������?A�������?I�������?a΂HgO�>i��Wo���?�Unknown
wfHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a΂HgO�>ia���?�Unknown
ygHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a΂HgO�>i�/��C��?�Unknown
�hHostReadVariableOp"*model_39/MoodOutput/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a΂HgO�>i5��Ix��?�Unknown
aiHostIdentity"Identity(1�������?9�������?A�������?I�������?a)��[�b�>iҵj���?�Unknown�
mjHostMul"RMSprop/RMSprop/update/mul_1(1�������?9�������?A�������?I�������?a)��[�b�>iom�����?�Unknown
mkHostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?a)��[�b�>i%l���?�Unknown
olHostAddV2"RMSprop/RMSprop/update/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��TPhv�>i��<�-��?�Unknown
umHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��TPhv�>i`ftV��?�Unknown
qnHostAddV2"RMSprop/RMSprop/update_1/add_1(1333333�?9333333�?A333333�?I333333�?a��D��>i�-�y��?�Unknown
moHostSub"RMSprop/RMSprop/update_1/sub(1333333�?9333333�?A333333�?I333333�?a��D��>i�yN����?�Unknown
upHostRealDiv" RMSprop/RMSprop/update_1/truediv(1333333�?9333333�?A333333�?I333333�?a��D��>i�o����?�Unknown
mqHostSub"RMSprop/RMSprop/update_2/sub(1333333�?9333333�?A333333�?I333333�?a��D��>i8������?�Unknown
�rHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?asX�rp;�>i�������?�Unknown*�j
uHostFlushSummaryWriter"FlushSummaryWriter(1    ��@9    ��@A    ��@I    ��@aNA����?iNA����?�Unknown�
iHostWriteSummary"WriteSummary(1ffffff6@9ffffff6@Affffff6@Iffffff6@a�B|��F?i�Q�~`��?�Unknown�
pHostSoftmax"model_39/MoodOutput/Softmax(1������5@9������5@A������5@I������5@a^��+F?il�Xh��?�Unknown
pHost_FusedMatMul"model_39/dense_39/Relu(13333334@93333334@A3333334@I3333334@a\��6ȻD?i۷fZ��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333�6@933333�6@A      3@I      3@aɾ��w�C?i�Z_x���?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(133333�1@933333�1@A33333�1@I33333�1@a�*y��*B?i�x�0���?�Unknown
~HostMatMul"*gradient_tape/model_39/MoodOutput/MatMul_1(1      0@9      0@A      0@I      0@aXB�/l@?i'`Z<���?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      .@9      .@A      .@I      .@af�F=��>?i	b�y��?�Unknown
�	HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff+@9ffffff+@Affffff+@Iffffff+@aд#1�<?iz-�����?�Unknown
|
HostMatMul"(gradient_tape/model_39/MoodOutput/MatMul(1������(@9������(@A������(@I������(@a�b���?9?i�`��%��?�Unknown
dHostDataset"Iterator::Model(1      =@9      =@A������&@I������&@a����h27?i�U����?�Unknown
\HostSub"RMSprop/sub(1333333&@9333333&@A333333&@I333333&@a�e�N�6?i�q_��?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1ffffff$@9ffffff$@Affffff$@Iffffff$@a��N�U�4?ig����?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1ffffff"@9ffffff"@Affffff"@Iffffff"@a?[���2?i��	f߼�?�Unknown
zHostMatMul"&gradient_tape/model_39/dense_39/MatMul(1      "@9      "@A      "@I      "@a���y2?i����.��?�Unknown
`HostGatherV2"
GatherV2_1(1������!@9������!@A������!@I������!@a6�+�'E2?id��Aw��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff @9ffffff @Affffff @Iffffff @a4�g�I�0?iW�����?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������(@9������(@A333333@I333333@a|��H0?i��]M���?�Unknown
�HostBiasAddGrad"5gradient_tape/model_39/MoodOutput/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a���臏-?i34�Ek��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a4S[R�,?if�7��?�Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@ac����+?iO������?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a<� %|t)?iX����?�Unknown�
�HostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a`86^a)?i������?�Unknown
_HostCast"model_39/Cast(1ffffff@9ffffff@Affffff@Iffffff@a`86^a)?i ܼ����?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�C@933333�C@A������@I������@a>� $?iħ�b���?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1������@9������@A������@I������@a>� $?ihs�D2��?�Unknown
`HostDivNoNan"
div_no_nan(1ffffff@9ffffff@Affffff@Iffffff@a?[���"?i��q`��?�Unknown
~HostReluGrad"(gradient_tape/model_39/dense_39/ReluGrad(1ffffff@9ffffff@Affffff@Iffffff@a?[���"?i�~�����?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a���y"?iߗ:���?�Unknown
ZHostArgMax"ArgMax(1������@9������@A������@I������@aȕ�*�"?iHD�C���?�Unknown
mHostMul"RMSprop/RMSprop/update_2/mul(1������@9������@A������@I������@aȕ�*�"?i��IM���?�Unknown
� HostBiasAddGrad"3gradient_tape/model_39/dense_39/BiasAdd/BiasAddGrad(1������@9������@A������@I������@aȕ�*�"?i��V��?�Unknown
v!HostCast"$sparse_categorical_crossentropy/Cast(1������@9������@A������@I������@aȕ�*�"?i�I�`:��?�Unknown
X"HostCast"Cast_3(1������@9������@A������@I������@a�1�d>!?i��FN��?�Unknown
m#HostMul"RMSprop/RMSprop/update_1/mul(1333333@9333333@A333333@I333333@a|��H ?iѩxN��?�Unknown
s$HostSquare"RMSprop/RMSprop/update_3/Square(1333333@9333333@A333333@I333333@a|��H ?i 7b�N��?�Unknown
u%Host_FusedMatMul"model_39/MoodOutput/BiasAdd(1333333@9333333@A333333@I333333@a|��H ?i/Ħ�N��?�Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff@9ffffff@Affffff@Iffffff@aA1�3?i��>zH��?�Unknown
�'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a��|v�a?i��*�;��?�Unknown
�(HostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1������@9������@A������@I������@a���臏?i��i(��?�Unknown
�)HostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1������@9������@A������@I������@a���臏?i'����?�Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��)@�?ij(�G���?�Unknown
o+HostMul"RMSprop/RMSprop/update_3/mul_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��)@�?i�)���?�Unknown
�,HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1������	@9������	@A������	@I������	@a�6���F?ic��D���?�Unknown
y-HostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1      @9      @A      @I      @a��k�F�?i�y�V]��?�Unknown
X.HostCast"Cast_2(1333333@9333333@A333333@I333333@a�9�	�?i|�g���?�Unknown
�/HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1333333@9333333@A333333@I333333@a�9�	�?i6�W���?�Unknown
o0HostAddV2"RMSprop/RMSprop/update_2/add(1333333@9333333@A333333@I333333@a�9�	�?i�exؘ��?�Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff@9ffffff@Affffff@Iffffff@a�B|��?iHT�P��?�Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_1(1������@9������@A������@I������@a^��+?it��$��?�Unknown
�3HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1������@9������@A������@I������@a^��+?i�2�����?�Unknown
�4HostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1333333@9333333@A333333@I333333@a7��E�?ibb�)Q��?�Unknown
o5HostMul"RMSprop/RMSprop/update_2/mul_2(1333333@9333333@A333333@I333333@a7��E�?i�����?�Unknown
o6HostSqrt"RMSprop/RMSprop/update_3/Sqrt(1333333@9333333@A333333@I333333@a7��E�?i`�1z���?�Unknown
o7HostSqrt"RMSprop/RMSprop/update_2/Sqrt(1ffffff@9ffffff@Affffff@Iffffff@a?[���?i:���#��?�Unknown
�8HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?[���?iG-����?�Unknown
V9HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a?[���?i�	��Q��?�Unknown
�:HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1������@9������@A������@I������@aȕ�*�?i#`|B���?�Unknown
X;HostEqual"Equal(1������ @9������ @A������ @I������ @a�1�d>?i�I�5l��?�Unknown
w<HostReadVariableOp"RMSprop/Cast/ReadVariableOp(1������ @9������ @A������ @I������ @a�1�d>?iA3�(���?�Unknown
�=HostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1������ @9������ @A������ @I������ @a�1�d>?i�����?�Unknown
s>HostSquare"RMSprop/RMSprop/update_2/Square(1������ @9������ @A������ @I������ @a�1�d>?i_
��?�Unknown
u?HostReadVariableOp"div_no_nan/ReadVariableOp(1������ @9������ @A������ @I������ @a�1�d>?i��4���?�Unknown
q@HostSquare"RMSprop/RMSprop/update/Square(1       @9       @A       @I       @aXB�/l?i�l�c��?�Unknown
�AHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @aXB�/l?i��%Ś��?�Unknown
TBHostMul"Mul(1ffffff�?9ffffff�?Affffff�?Iffffff�?aA1�3?i����?�Unknown
yCHostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aA1�3?iL
�d���?�Unknown
�DHostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aA1�3?i��4��?�Unknown
�EHostReadVariableOp"'model_39/dense_39/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aA1�3?i�*V���?�Unknown
oFHostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?a���臏?iu�uB��?�Unknown
�GHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1�������?9�������?A�������?I�������?a���臏?ir��z��?�Unknown
qHHostAddV2"RMSprop/RMSprop/update_3/add_1(1�������?9�������?A�������?I�������?a���臏?i������?�Unknown
uIHostRealDiv" RMSprop/RMSprop/update_3/truediv(1�������?9�������?A�������?I�������?a���臏?iR���f��?�Unknown
�JHostReadVariableOp")model_39/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a���臏?i�\�:���?�Unknown
�KHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1333333�?9333333�?A333333�?I333333�?ac����?i�g�L��?�Unknown
�LHostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1�������?9�������?A�������?I�������?a�6���F
?i@^.���?�Unknown
�MHostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1      �?9      �?A      �?I      �?a��k�F�?i�H���?�Unknown
mNHostAddV2"RMSprop/RMSprop/update/add(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�B|��?i����t��?�Unknown
mOHostMul"RMSprop/RMSprop/update/mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�B|��?i�$z���?�Unknown
�PHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�B|��?i��q,��?�Unknown
�QHostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a�<apY?isTׁ��?�Unknown
kRHostSub"RMSprop/RMSprop/update/sub(1�������?9�������?A�������?I�������?a�<apY?iؗ=���?�Unknown
oSHostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?a�<apY?i=ע,��?�Unknown
wTHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�<apY?i������?�Unknown
�UHostReadVariableOp"(model_39/dense_39/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�<apY?i%Zn���?�Unknown
�VHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1333333�?9333333�?A333333�?I333333�?a7��E�?i�<oB&��?�Unknown
oWHostSqrt"RMSprop/RMSprop/update_1/Sqrt(1333333�?9333333�?A333333�?I333333�?a7��E�?i�T�u��?�Unknown
�XHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a7��E�?iGl�����?�Unknown
oYHostAddV2"RMSprop/RMSprop/update_3/add(1333333�?9333333�?A333333�?I333333�?a7��E�?i�����?�Unknown
oZHostMul"RMSprop/RMSprop/update_3/mul_1(1333333�?9333333�?A333333�?I333333�?a7��E�?iǛÒa��?�Unknown
b[HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a7��E�?i���f���?�Unknown
�\HostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?aXB�/l ?i�����?�Unknown
�]HostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?aXB�/l ?iq0Q�3��?�Unknown
m^HostSqrt"RMSprop/RMSprop/update/Sqrt(1�������?9�������?A�������?I�������?a���臏�>iAa�n��?�Unknown
s_HostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?a���臏�>i�p���?�Unknown
s`HostSquare"RMSprop/RMSprop/update_1/Square(1�������?9�������?A�������?I�������?a���臏�>i᥀%���?�Unknown
oaHostAddV2"RMSprop/RMSprop/update_1/add(1�������?9�������?A�������?I�������?a���臏�>i�w�D ��?�Unknown
obHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a���臏�>i�I�c[��?�Unknown
qcHostAddV2"RMSprop/RMSprop/update_2/add_1(1�������?9�������?A�������?I�������?a���臏�>iQ�����?�Unknown
wdHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a���臏�>i!������?�Unknown
yeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a���臏�>i�����?�Unknown
�fHostReadVariableOp"*model_39/MoodOutput/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a���臏�>i����G��?�Unknown
agHostIdentity"Identity(1�������?9�������?A�������?I�������?a�6���F�>i��Bm|��?�Unknown�
mhHostMul"RMSprop/RMSprop/update/mul_1(1�������?9�������?A�������?I�������?a�6���F�>i[�����?�Unknown
miHostSub"RMSprop/RMSprop/update_3/sub(1�������?9�������?A�������?I�������?a�6���F�>i?�	����?�Unknown
ojHostAddV2"RMSprop/RMSprop/update/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�B|���>iĸ����?�Unknown
ukHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�B|���>iI�wA��?�Unknown
qlHostAddV2"RMSprop/RMSprop/update_1/add_1(1333333�?9333333�?A333333�?I333333�?a7��E��>i)=��h��?�Unknown
mmHostSub"RMSprop/RMSprop/update_1/sub(1333333�?9333333�?A333333�?I333333�?a7��E��>i	ɌS���?�Unknown
unHostRealDiv" RMSprop/RMSprop/update_1/truediv(1333333�?9333333�?A333333�?I333333�?a7��E��>i�T�����?�Unknown
moHostSub"RMSprop/RMSprop/update_2/sub(1333333�?9333333�?A333333�?I333333�?a7��E��>i��'���?�Unknown
�pHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1      �?9      �?A      �?I      �?aXB�/l�>i     �?�Unknown2GPU