function NeuralStructure = ReadDoric(Dataname)
%%
tic
fprintf('Read Doric data to table \n')
T = readtable(Dataname);
names = T.Properties.VariableNames;

% save Trigger in orignal Sampling rate
T_CamTrigONind = find(diff([0;T.(names{5})])==1);
T_CamTrigOFFind = find(diff([0;T.(names{5})])==-1);
T_CamTrigON = T.(names{1})(T_CamTrigONind);
T_CamTrigOFF = T.(names{1})(T_CamTrigOFFind);

if numel(T_CamTrigON) < 100
    T_CamTrigONind = find(diff([0;T.(names{7})])==1);
    T_CamTrigOFFind = find(diff([0;T.(names{7})])==-1);
    T_CamTrigON = T.(names{1})(T_CamTrigONind);
    T_CamTrigOFF = T.(names{1})(T_CamTrigOFFind);
end

NeuralStructure = struct();
NeuralStructure.Time = T.(names{1});
NeuralStructure.Chan_Iso = T.(names{2});
NeuralStructure.Chan_GCamP = T.(names{3});

NeuralStructure.T_CamTrigON = T_CamTrigON;
NeuralStructure.T_CamTrigOFF = T_CamTrigOFF;

toc
end