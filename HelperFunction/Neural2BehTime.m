function InterpBehStrcture = Neural2BehTime(BehStrcture,NeuralTriggerON)
InterpBehStrcture = struct();

%% trim tiggers
BehTriggerON = BehStrcture.CamTrigON;
BehTriggerON = BehTriggerON(1:min(numel(BehTriggerON),numel(NeuralTriggerON)));
NeuralTriggerON = NeuralTriggerON(1:min(numel(BehTriggerON),numel(NeuralTriggerON)));

InterpBehStrcture.BehTriggerON = BehTriggerON;
InterpBehStrcture.NeuralTriggerON = NeuralTriggerON;

%% interpolation
names = fieldnames(BehStrcture);
for i = 1:length(names)
    InterpBehStrcture.(names{i}) = interp1(BehTriggerON,NeuralTriggerON,BehStrcture.(names{i}),'linear','extrap');
end

end