"�t
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(13333s-�@93333s-�@A3333s-�@I3333s-�@aǏG�O�?iǏG�O�?�Unknown�
BHostIDLE"IDLE13333���@A3333���@a�vђI��?i���y��?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(133333�c@933333�c@A����̬a@I����̬a@a��yw�ib?i�oa.]��?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1333333]@9333333]@A33333R@I33333R@a��KIa�R?i�_Ǟ�?�Unknown
tHost_FusedMatMul"model_2/MoodOutput/BiasAdd(1     �N@9     �N@A     �N@I     �N@a�d�~��O?i8�eݸ��?�Unknown
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1     @F@9     @F@A33333�D@I33333�D@a�
cm�E?i�%A�-��?�Unknown
xHostMatMul"$gradient_tape/model_2/dense_2/MatMul(1     �B@9     �B@A     �B@I     �B@a�Z;�EC?i�4{���?�Unknown
n	Host_FusedMatMul"model_2/dense_2/Relu(1fffff�>@9fffff�>@Afffff�>@Ifffff�>@asPX(S@?i�JE'��?�Unknown
o
HostSoftmax"model_2/MoodOutput/Softmax(1      =@9      =@A      =@I      =@a�
l�5>?i&�r�˸�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(133333�<@933333�<@A33333�<@I33333�<@azV����=?i��l����?�Unknown
{HostMatMul"'gradient_tape/model_2/MoodOutput/MatMul(1������7@9������7@A������7@I������7@a��\�*�8?iP-����?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������7@9������7@A������7@I������7@a��s[ԕ8?i˛Pô��?�Unknown
}HostMatMul")gradient_tape/model_2/MoodOutput/MatMul_1(13333337@93333337@A3333337@I3333337@an���'+8?i��A(���?�Unknown
�HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������5@9������5@A������5@I������5@a*�A�ʵ6?i88����?�Unknown
mHostMul"RMSprop/RMSprop/update_1/mul(1     �5@9     �5@A     �5@I     �5@a��e6?i����]��?�Unknown
iHostWriteSummary"WriteSummary(133333�4@933333�4@A33333�4@I33333�4@a�?jo�5?i�<����?�Unknown�
sHostSquare"RMSprop/RMSprop/update_1/Square(1������2@9������2@A������2@I������2@a��d`3?i��)�{��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1     �0@9     �0@A     �0@I     �0@a&. �X01?i��C����?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1������+@9������+@A������+@I������+@aOO����,?i��q��?�Unknown
^HostCast"model_2/Cast(1      *@9      *@A      *@I      *@a�@aF�+?i��}"��?�Unknown
\HostSub"RMSprop/sub(1������)@9������)@A������)@I������)@a�"x݊�*?i]�����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      1@9      1@A      )@I      )@aƪ�91*?i���8q��?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1      (@9      (@A      (@I      (@a�F-� )?i��@��?�Unknown
mHostMul"RMSprop/RMSprop/update_2/mul(1ffffff&@9ffffff&@Affffff&@Iffffff&@aF$���U'?i�.��v��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������$@9������$@A������$@I������$@a�3���%?i0dO���?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������$@9������$@A������$@I������$@a�3���%?is,��?�Unknown
hHostRandomShuffle"RandomShuffle(1������#@9������#@A������#@I������#@aȝ&�j�$?i�'�v��?�Unknown
�HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1������"@9������"@A������"@I������"@a�����#?im�_c���?�Unknown
dHostDataset"Iterator::Model(1������B@9������B@A������!@I������!@afqy
�"?i$���?�Unknown
|HostReluGrad"&gradient_tape/model_2/dense_2/ReluGrad(1ffffff @9ffffff @Affffff @Iffffff @a!����!?i޼�n���?�Unknown
^ HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@al�N1�?iSGR���?�Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1����̬f@9����̬f@A333333@I333333@a1�׻�U?i&�͸��?�Unknown
r"HostTensorSliceDataset"TensorSliceDataset(1������@9������@A������@I������@a��:�?iAvg'���?�Unknown
�#HostBiasAddGrad"1gradient_tape/model_2/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a��t4�?i��m��?�Unknown
e$Host
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@aZ`Ϸz�?i4���+��?�Unknown�
�%HostBiasAddGrad"4gradient_tape/model_2/MoodOutput/BiasAdd/BiasAddGrad(1������@9������@A������@I������@aZ`Ϸz�?i�������?�Unknown
Z&HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@aF$���U?i��%7���?�Unknown
�'HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1333333@9333333@A333333@I333333@ap�p�?i�Ka�T��?�Unknown
�(HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1333333@9333333@A333333@I333333@ap�p�?i М���?�Unknown
v)HostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@ap�p�?i4T�A���?�Unknown
`*HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a�3���?i�I��c��?�Unknown
�+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @ap���`�?izY�����?�Unknown
�,HostReadVariableOp"%model_2/dense_2/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@aHP>�?i�K�����?�Unknown
o-HostSqrt"RMSprop/RMSprop/update_1/Sqrt(1������@9������@A������@I������@a5�}lZ�?i�����?�Unknown
�.HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a!����?iG�.����?�Unknown
m/HostMul"RMSprop/RMSprop/update/mul_2(1      @9      @A      @I      @ac�� �?i�4�"��?�Unknown
T0HostMul"Mul(1������@9������@A������@I������@a�� � ?i�ؠ ���?�Unknown
q1HostSquare"RMSprop/RMSprop/update/Square(1������@9������@A������@I������@a�� � ?i���?�Unknown
�2HostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1      @9      @A      @I      @aXm|_A+?ic����?�Unknown
�3HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff
@9ffffff
@Affffff
@Iffffff
@a
}3��?i��J����?�Unknown
b4HostDivNoNan"div_no_nan_1(1������@9������@A������@I������@a������	?i��	]��?�Unknown
�5HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1������@9������@A������@I������@a������	?i.K!a���?�Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a�F-� 	?iF &c(��?�Unknown
X7HostEqual"Equal(1      @9      @A      @I      @a�F-� 	?i^�*e���?�Unknown
�8HostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1      @9      @A      @I      @a�F-� 	?ivj/g���?�Unknown
o9HostMul"RMSprop/RMSprop/update_2/mul_2(1      @9      @A      @I      @a�F-� 	?i�4iT��?�Unknown
t:HostAssignAddVariableOp"AssignAddVariableOp(1333333@9333333@A333333@I333333@an���'+?iF����?�Unknown
�;HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1333333@9333333@A333333@I333333@an���'+?i�lp���?�Unknown
X<HostCast"Cast_2(1ffffff@9ffffff@Affffff@Iffffff@aF$���U?i��s��?�Unknown
q=HostAddV2"RMSprop/RMSprop/update_3/add_1(1ffffff@9ffffff@Affffff@Iffffff@aF$���U?i���p���?�Unknown
�>HostReadVariableOp")model_2/MoodOutput/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@aF$���U?iy4�-��?�Unknown
y?HostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1������@9������@A������@I������@a �XBt�?i�=�ɇ��?�Unknown
X@HostSlice"Slice(1������@9������@A������@I������@a �XBt�?i?G�����?�Unknown
�AHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1������@9�������?A������@I�������?a�3���?i�#x8��?�Unknown
�BHostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1������@9������@A������@I������@a�3���?i�<�$���?�Unknown
XCHostCast"Cast_3(1      @9      @A      @I      @aѻ���?i )�{���?�Unknown
�DHostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1      @9      @A      @I      @aѻ���?i_��5��?�Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_4(1333333@9333333@A333333@I333333@a�CkWg ?is3ԅ��?�Unknown
oFHostSqrt"RMSprop/RMSprop/update_2/Sqrt(1333333@9333333@A333333@I333333@a�CkWg ?i�������?�Unknown
wGHostReadVariableOp"RMSprop/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a��Ƴ+?iԟ�"��?�Unknown
�HHostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a��Ƴ+?i�n>.o��?�Unknown
sIHostSquare"RMSprop/RMSprop/update_2/Square(1������@9������@A������@I������@a]S"�U?ix�����?�Unknown
yJHostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1������ @9������ @A������ @I������ @a5�}lZ�?ioax����?�Unknown
oKHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��iJN��>iC��=��?�Unknown
�LHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��iJN��>i��3}��?�Unknown
sMHostSquare"RMSprop/RMSprop/update_3/Square(1�������?9�������?A�������?I�������?a�� � �>iY��4���?�Unknown
mNHostSqrt"RMSprop/RMSprop/update/Sqrt(1333333�?9333333�?A333333�?I333333�?a1�׻�U�>i		�����?�Unknown
�OHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1�������?9�������?A�������?I�������?a��t4��>i'�7'��?�Unknown
�PHostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a��t4��>iEۈ�\��?�Unknown
VQHostSum"Sum_2(1�������?9�������?A�������?I�������?a��t4��>ic�����?�Unknown
�RHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      �?9      �?A      �?I      �?a�F-� �>i������?�Unknown
mSHostAddV2"RMSprop/RMSprop/update/add(1      �?9      �?A      �?I      �?a�F-� �>i{y�����?�Unknown
oTHostAddV2"RMSprop/RMSprop/update_2/add(1      �?9      �?A      �?I      �?a�F-� �>i���'��?�Unknown
oUHostMul"RMSprop/RMSprop/update_3/mul_2(1      �?9      �?A      �?I      �?a�F-� �>i�.��Y��?�Unknown
yVHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a�F-� �>i�����?�Unknown
uWHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?aF$���U�>iU�����?�Unknown
oXHostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?a�3����>i�������?�Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�CkWg �>iXA����?�Unknown
�ZHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�CkWg �>i/�k�5��?�Unknown
�[HostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�CkWg �>i�:�]��?�Unknown
`\HostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a�CkWg �>i�M	���?�Unknown
v]HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a]S"�U�>i"nq����?�Unknown
�^HostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1�������?9�������?A�������?I�������?a]S"�U�>ig��D���?�Unknown
k_HostSub"RMSprop/RMSprop/update/sub(1�������?9�������?A�������?I�������?a]S"�U�>i��A����?�Unknown
s`HostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?a]S"�U�>i�Ω���?�Unknown
�aHostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1�������?9�������?A�������?I�������?a]S"�U�>i6�G=��?�Unknown
�bHostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a]S"�U�>i{z�a��?�Unknown
ocHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a]S"�U�>i�/❆��?�Unknown
mdHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a]S"�U�>iPJI���?�Unknown
oeHostAddV2"RMSprop/RMSprop/update_3/add(1�������?9�������?A�������?I�������?a]S"�U�>iJp�����?�Unknown
�fHostReadVariableOp"(model_2/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a]S"�U�>i������?�Unknown
�gHostReadVariableOp"&model_2/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a]S"�U�>i԰�K��?�Unknown
mhHostMul"RMSprop/RMSprop/update/mul_1(1      �?9      �?A      �?I      �?ac�� ��>i�B��:��?�Unknown
�iHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1      �?9      �?A      �?I      �?ac�� ��>i:ԅ�[��?�Unknown
ojHostMul"RMSprop/RMSprop/update_3/mul_1(1      �?9      �?A      �?I      �?ac�� ��>i�e�M}��?�Unknown
okHostAddV2"RMSprop/RMSprop/update_1/add(1�������?9�������?A�������?I�������?a�� � �>ii"N���?�Unknown
mlHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a�� � �>i/l�N���?�Unknown
�mHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a�� � �>iPoXO���?�Unknown
anHostIdentity"Identity(1�������?9�������?A�������?I�������?a��t4��>i������?�Unknown�
ooHostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?a��t4��>inX����?�Unknown
upHostRealDiv" RMSprop/RMSprop/update_1/truediv(1�������?9�������?A�������?I�������?a��t4��>i���P'��?�Unknown
�qHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aF$���U�>i��æ>��?�Unknown
qrHostAddV2"RMSprop/RMSprop/update_1/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aF$���U�>i����U��?�Unknown
qsHostAddV2"RMSprop/RMSprop/update_2/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aF$���U�>i�~_Rm��?�Unknown
�tHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aF$���U�>i�d-����?�Unknown
uuHostRealDiv" RMSprop/RMSprop/update_3/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?aF$���U�>i�J�����?�Unknown
wvHostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aF$���U�>i�0�S���?�Unknown
mwHostSub"RMSprop/RMSprop/update_3/sub(1333333�?9333333�?A333333�?I333333�?a�CkWg �>iV�0T���?�Unknown
uxHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�CkWg �>i�ߗT���?�Unknown
wyHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�CkWg �>i,7�T���?�Unknown
ozHostAddV2"RMSprop/RMSprop/update/add_1(1      �?9      �?A      �?I      �?ac�� ��>i     �?�Unknown*�r
uHostFlushSummaryWriter"FlushSummaryWriter(13333s-�@93333s-�@A3333s-�@I3333s-�@aq�Ru�?iq�Ru�?�Unknown�
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(133333�c@933333�c@A����̬a@I����̬a@a��	YUud?i��q����?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1333333]@9333333]@A33333R@I33333R@aߪ2���T?iO�X����?�Unknown
tHost_FusedMatMul"model_2/MoodOutput/BiasAdd(1     �N@9     �N@A     �N@I     �N@a�+�2ƦQ?ieSr�͜�?�Unknown
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1     @F@9     @F@A33333�D@I33333�D@a_NJ8�?H?i�e �ݢ�?�Unknown
xHostMatMul"$gradient_tape/model_2/dense_2/MatMul(1     �B@9     �B@A     �B@I     �B@a�^���iE?i��[8��?�Unknown
nHost_FusedMatMul"model_2/dense_2/Relu(1fffff�>@9fffff�>@Afffff�>@Ifffff�>@a��Tm	�A?i�fް��?�Unknown
oHostSoftmax"model_2/MoodOutput/Softmax(1      =@9      =@A      =@I      =@a��@?i�+� ��?�Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(133333�<@933333�<@A33333�<@I33333�<@aa�#��@?ik�v
��?�Unknown
{
HostMatMul"'gradient_tape/model_2/MoodOutput/MatMul(1������7@9������7@A������7@I������7@a�_;@�;?iWV~�{��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������7@9������7@A������7@I������7@a��z �P;?i�e���?�Unknown
}HostMatMul")gradient_tape/model_2/MoodOutput/MatMul_1(13333337@93333337@A3333337@I3333337@a�Q�v�:?i����@��?�Unknown
�HostAssignVariableOp"'RMSprop/RMSprop/update/AssignVariableOp(1������5@9������5@A������5@I������5@a]�A�;9?i��ph��?�Unknown
mHostMul"RMSprop/RMSprop/update_1/mul(1     �5@9     �5@A     �5@I     �5@a��b��8?iz$QȄ��?�Unknown
iHostWriteSummary"WriteSummary(133333�4@933333�4@A33333�4@I33333�4@a2�/��7?i�~���?�Unknown�
sHostSquare"RMSprop/RMSprop/update_1/Square(1������2@9������2@A������2@I������2@aW��Gl�5?i��k4��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1     �0@9     �0@A     �0@I     �0@az��`*3?i�!쐗��?�Unknown
mHostMul"RMSprop/RMSprop/update_3/mul(1������+@9������+@A������+@I������+@aEYUg�0?id�h���?�Unknown
^HostCast"model_2/Cast(1      *@9      *@A      *@I      *@a�q�#.?i~5�{��?�Unknown
\HostSub"RMSprop/sub(1������)@9������)@A������)@I������)@a%	݄��-?iOQ=�Y��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      1@9      1@A      )@I      )@adŊ���,?i��v�(��?�Unknown
kHostMul"RMSprop/RMSprop/update/mul(1      (@9      (@A      (@I      (@a��u��+?i:T�.���?�Unknown
mHostMul"RMSprop/RMSprop/update_2/mul(1ffffff&@9ffffff&@Affffff&@Iffffff&@a.i��i�)?i1dH���?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������$@9������$@A������$@I������$@a��Z�O(?i�)E:��?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������$@9������$@A������$@I������$@a��Z�O(?i��Ao���?�Unknown
hHostRandomShuffle"RandomShuffle(1������#@9������#@A������#@I������#@a�t���&?i�f<���?�Unknown
�HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1������"@9������"@A������"@I������"@aG8����%?i��4JQ��?�Unknown
dHostDataset"Iterator::Model(1������B@9������B@A������!@I������!@a�c�]_�$?ij*���?�Unknown
|HostReluGrad"&gradient_tape/model_2/dense_2/ReluGrad(1ffffff @9ffffff @Affffff @Iffffff @am�È�"?io������?�Unknown
^HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a�24?�o ?i��
����?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1����̬f@9����̬f@A333333@I333333@a����{?i�À���?�Unknown
r HostTensorSliceDataset"TensorSliceDataset(1������@9������@A������@I������@a��é0?i9�H����?�Unknown
�!HostBiasAddGrad"1gradient_tape/model_2/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a5xHJ��?i}.3����?�Unknown
e"Host
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a�(�c?i�߳օ��?�Unknown�
�#HostBiasAddGrad"4gradient_tape/model_2/MoodOutput/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�(�c?i�4�X��?�Unknown
Z$HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a.i��i�?i ��a(��?�Unknown
�%HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice(1333333@9333333@A333333@I333333@a��A։?i �3����?�Unknown
�&HostReadVariableOp"%RMSprop/RMSprop/update/ReadVariableOp(1333333@9333333@A333333@I333333@a��A։?i@������?�Unknown
v'HostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@a��A։?i`��Mu��?�Unknown
`(HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a��Z�O?i7 �5��?�Unknown
�)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a��:���?i�*����?�Unknown
�*HostReadVariableOp"%model_2/dense_2/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@að譕�?iUQ��{��?�Unknown
o+HostSqrt"RMSprop/RMSprop/update_1/Sqrt(1������@9������@A������@I������@a㎿8r?iQRj��?�Unknown
�,HostReadVariableOp"+RMSprop/RMSprop/update_2/mul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@am�È�?i4�F���?�Unknown
m-HostMul"RMSprop/RMSprop/update/mul_2(1      @9      @A      @I      @a!KmN�?in��nC��?�Unknown
T.HostMul"Mul(1������@9������@A������@I������@a���y�?i�u�����?�Unknown
q/HostSquare"RMSprop/RMSprop/update/Square(1������@9������@A������@I������@a���y�?i�C2N��?�Unknown
�0HostReadVariableOp"+RMSprop/RMSprop/update_1/mul/ReadVariableOp(1      @9      @A      @I      @a���b4?i�hB����?�Unknown
�1HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff
@9ffffff
@Affffff
@Iffffff
@a���4��?ib;��I��?�Unknown
b2HostDivNoNan"div_no_nan_1(1������@9������@A������@I������@as4�_��?i;�,ͼ��?�Unknown
�3HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1������@9������@A������@I������@as4�_��?i;n�/��?�Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a��u��?i�|����?�Unknown
X5HostEqual"Equal(1      @9      @A      @I      @a��u��?i4����?�Unknown
�6HostReadVariableOp"'RMSprop/RMSprop/update_1/ReadVariableOp(1      @9      @A      @I      @a��u��?iľ��|��?�Unknown
o7HostMul"RMSprop/RMSprop/update_2/mul_2(1      @9      @A      @I      @a��u��?iT�����?�Unknown
t8HostAssignAddVariableOp"AssignAddVariableOp(1333333@9333333@A333333@I333333@a�Q�v�
?i���W��?�Unknown
�9HostAssignVariableOp")RMSprop/RMSprop/update_3/AssignVariableOp(1333333@9333333@A333333@I333333@a�Q�v�
?i��Y����?�Unknown
X:HostCast"Cast_2(1ffffff@9ffffff@Affffff@Iffffff@a.i��i�	?i�s �*��?�Unknown
q;HostAddV2"RMSprop/RMSprop/update_3/add_1(1ffffff@9ffffff@Affffff@Iffffff@a.i��i�	?i���V���?�Unknown
�<HostReadVariableOp")model_2/MoodOutput/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a.i��i�	?i�{M���?�Unknown
y=HostReadVariableOp"RMSprop/Cast_1/ReadVariableOp(1������@9������@A������@I������@am%��\ 	?i�V�^��?�Unknown
X>HostSlice"Slice(1������@9������@A������@I������@am%��\ 	?iF13���?�Unknown
�?HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1������@9�������?A������@I�������?a��Z�O?i�br\"��?�Unknown
�@HostReadVariableOp"'RMSprop/RMSprop/update_3/ReadVariableOp(1������@9������@A������@I������@a��Z�O?i������?�Unknown
XAHostCast"Cast_3(1      @9      @A      @I      @a��B&?i@�B���?�Unknown
�BHostAssignVariableOp")RMSprop/RMSprop/update_2/AssignVariableOp(1      @9      @A      @I      @a��B&?ib���;��?�Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_4(1333333@9333333@A333333@I333333@a'Z��59?i;������?�Unknown
oDHostSqrt"RMSprop/RMSprop/update_2/Sqrt(1333333@9333333@A333333@I333333@a'Z��59?ibx����?�Unknown
wEHostReadVariableOp"RMSprop/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@afd)L?i���B��?�Unknown
�FHostAssignVariableOp")RMSprop/RMSprop/update_1/AssignVariableOp(1ffffff@9ffffff@Affffff@Iffffff@afd)L?i4�����?�Unknown
sGHostSquare"RMSprop/RMSprop/update_2/Square(1������@9������@A������@I������@a��#_?i{Y1����?�Unknown
yHHostReadVariableOp"RMSprop/Cast_2/ReadVariableOp(1������ @9������ @A������ @I������ @a㎿8r?iy<nK7��?�Unknown
oIHostSqrt"RMSprop/RMSprop/update_3/Sqrt(1ffffff�?9ffffff�?Affffff�?Iffffff�?a_d��?i��C�}��?�Unknown
�JHostReadVariableOp"+RMSprop/RMSprop/update_3/mul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a_d��?iQ]���?�Unknown
sKHostSquare"RMSprop/RMSprop/update_3/Square(1�������?9�������?A�������?I�������?a���y� ?itD����?�Unknown
mLHostSqrt"RMSprop/RMSprop/update/Sqrt(1333333�?9333333�?A333333�?I333333�?a����{�>iN�)�E��?�Unknown
�MHostAssignVariableOp")RMSprop/RMSprop/update/AssignVariableOp_1(1�������?9�������?A�������?I�������?a5xHJ���>i�d���?�Unknown
�NHostReadVariableOp")RMSprop/RMSprop/update/mul/ReadVariableOp(1�������?9�������?A�������?I�������?a5xHJ���>ip��4���?�Unknown
VOHostSum"Sum_2(1�������?9�������?A�������?I�������?a5xHJ���>i@�w���?�Unknown
�PHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      �?9      �?A      �?I      �?a��u���>iI+�/��?�Unknown
mQHostAddV2"RMSprop/RMSprop/update/add(1      �?9      �?A      �?I      �?a��u���>i��f��?�Unknown
oRHostAddV2"RMSprop/RMSprop/update_2/add(1      �?9      �?A      �?I      �?a��u���>i��$���?�Unknown
oSHostMul"RMSprop/RMSprop/update_3/mul_2(1      �?9      �?A      �?I      �?a��u���>i!������?�Unknown
yTHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a��u���>ii��B��?�Unknown
uUHostRealDiv" RMSprop/RMSprop/update_2/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a.i��i��>ih�A��?�Unknown
oVHostMul"RMSprop/RMSprop/update_2/mul_1(1�������?9�������?A�������?I�������?a��Z�O�>i�nDq��?�Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a'Z��59�>i��ڶ���?�Unknown
�XHostReadVariableOp"'RMSprop/RMSprop/update_2/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a'Z��59�>i��F)���?�Unknown
�YHostAssignVariableOp"+RMSprop/RMSprop/update_3/AssignVariableOp_1(1333333�?9333333�?A333333�?I333333�?a'Z��59�>ie������?�Unknown
`ZHostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a'Z��59�>i�p#��?�Unknown
v[HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a��#_�>i��V�K��?�Unknown
�\HostAssignAddVariableOp"#RMSprop/RMSprop/AssignAddVariableOp(1�������?9�������?A�������?I�������?a��#_�>i���t��?�Unknown
k]HostSub"RMSprop/RMSprop/update/sub(1�������?9�������?A�������?I�������?a��#_�>i>C�H���?�Unknown
s^HostRealDiv"RMSprop/RMSprop/update/truediv(1�������?9�������?A�������?I�������?a��#_�>ib�����?�Unknown
�_HostAssignVariableOp"+RMSprop/RMSprop/update_1/AssignVariableOp_1(1�������?9�������?A�������?I�������?a��#_�>i��7����?�Unknown
�`HostReadVariableOp",RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a��#_�>i�p���?�Unknown
oaHostMul"RMSprop/RMSprop/update_1/mul_1(1�������?9�������?A�������?I�������?a��#_�>i�[�A@��?�Unknown
mbHostSub"RMSprop/RMSprop/update_2/sub(1�������?9�������?A�������?I�������?a��#_�>i���h��?�Unknown
ocHostAddV2"RMSprop/RMSprop/update_3/add(1�������?9�������?A�������?I�������?a��#_�>i�����?�Unknown
�dHostReadVariableOp"(model_2/MoodOutput/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��#_�>i:.Q|���?�Unknown
�eHostReadVariableOp"&model_2/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��#_�>i^t�:���?�Unknown
mfHostMul"RMSprop/RMSprop/update/mul_1(1      �?9      �?A      �?I      �?a!KmN��>i9�D��?�Unknown
�gHostAssignVariableOp"+RMSprop/RMSprop/update_2/AssignVariableOp_1(1      �?9      �?A      �?I      �?a!KmN��>i��N-��?�Unknown
ohHostMul"RMSprop/RMSprop/update_3/mul_1(1      �?9      �?A      �?I      �?a!KmN��>i�J�XR��?�Unknown
oiHostAddV2"RMSprop/RMSprop/update_1/add(1�������?9�������?A�������?I�������?a���y��>i�>h�s��?�Unknown
mjHostSub"RMSprop/RMSprop/update_1/sub(1�������?9�������?A�������?I�������?a���y��>i29���?�Unknown
�kHostReadVariableOp",RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp(1�������?9�������?A�������?I�������?a���y��>i�%
Z���?�Unknown
alHostIdentity"Identity(1�������?9�������?A�������?I�������?a5xHJ���>i�o�����?�Unknown�
omHostMul"RMSprop/RMSprop/update_1/mul_2(1�������?9�������?A�������?I�������?a5xHJ���>i5�D����?�Unknown
unHostRealDiv" RMSprop/RMSprop/update_1/truediv(1�������?9�������?A�������?I�������?a5xHJ���>i}�>��?�Unknown
�oHostReadVariableOp"*RMSprop/RMSprop/update/Sqrt/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a.i��i��>i|�K,)��?�Unknown
qpHostAddV2"RMSprop/RMSprop/update_1/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a.i��i��>i{F�C��?�Unknown
qqHostAddV2"RMSprop/RMSprop/update_2/add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a.i��i��>iz�]��?�Unknown
�rHostReadVariableOp",RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a.i��i��>iy���v��?�Unknown
usHostRealDiv" RMSprop/RMSprop/update_3/truediv(1ffffff�?9ffffff�?Affffff�?Iffffff�?a.i��i��>ix)����?�Unknown
wtHostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a.i��i��>iw�[Ϫ��?�Unknown
muHostSub"RMSprop/RMSprop/update_3/sub(1333333�?9333333�?A333333�?I333333�?a'Z��59�>i-���?�Unknown
uvHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a'Z��59�>i��A���?�Unknown
wwHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a'Z��59�>i���z���?�Unknown
oxHostAddV2"RMSprop/RMSprop/update/add_1(1      �?9      �?A      �?I      �?a!KmN��>i     �?�Unknown2GPU