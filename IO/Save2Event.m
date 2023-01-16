function Save2Event(DataStrcture,savepath,NeuralTriggerOn)
names = fieldnames(DataStrcture);
allNames = [];
allONs = [];
allOFFs = [];
for i = 1:length(names)
    if ~isempty(strfind(names{i},'ON'))
        allNames = [allNames; repmat(string(names{i}(1:end-2)),length(DataStrcture.(names{i})),1)];
        allONs = [allONs; DataStrcture.(names{i})];
        allOFFs = [allOFFs; DataStrcture.([names{i}(1:end-2),'OFF'])];
    end
end
[~,b] = sort(allONs);
allNames = allNames(b);
allONs = allONs(b);
allOFFs = allOFFs(b);

%% Map beh timestamps from beh time space to Neural time space
BehTriggerON = DataStrcture.CamTrigON;
BehTriggerON = BehTriggerON(1:min(numel(BehTriggerON),numel(NeuralTriggerOn)));
NeuralTriggerOn = NeuralTriggerOn(1:min(numel(BehTriggerON),numel(NeuralTriggerOn)));

allONs_interp = interp1(BehTriggerON,NeuralTriggerOn,allONs,'linear','extrap');
allOFFs_interp = interp1(BehTriggerON,NeuralTriggerOn,allOFFs,'linear','extrap');

% check again if ON always earlier than OFF
timeDelays = allONs_interp - allOFFs_interp;
if any(timeDelays>0)
    msg = 'Some ONs is latter than their OFFs';
    error(msg)
end
if ~issorted(allONs_interp)
    msg = 'ON is not sorted';
    error(msg)
end

T = table(allNames,allONs_interp,allOFFs_interp, 'VariableNames',{'Event','Onset','Offset'});

%% save csv
a = strfind(savepath,'/20');
filename = savepath(a+1:end-1);
savename = [savepath,filename,'_Events.csv'];
fprintf('Save file %s in %s \n', [filename,'_Events.csv'], savepath)
writetable(T,savename,'Delimiter',',','QuoteStrings',true);
fprintf('done ! \n')