function [T_CamTrigON, T_CamTrigOFF] = Save2Doric(Dataname,savepath)
%%
tic
fprintf('Read Doric data to table \n')
T = readtable(Dataname);

% save Trigger in orignal Sampling rate
T_CamTrigONind = find(diff([0;T.DigitalI_O_Ch_1])==1);
T_CamTrigOFFind = find(diff([0;T.DigitalI_O_Ch_1])==-1);
T_CamTrigON = T.x___(T_CamTrigONind);
T_CamTrigOFF = T.x___(T_CamTrigOFFind);

% down sampling from 12kHz to 120Hz (100 times)
% first change nans to zeros
AnalogIn__Ch_1 = T.AnalogIn__Ch_1;
AnalogIn__Ch_1_1 = T.AnalogIn__Ch_1_1;
AnalogIn__Ch_1 = fillmissing(T.AnalogIn__Ch_1,'nearest');
AnalogIn__Ch_1_1 = fillmissing(T.AnalogIn__Ch_1_1,'nearest');

r = 100;
T_time = downsample(T.x___,r);
T_Chan_Iso = downsample(AnalogIn__Ch_1,r);
T_Chan_GCamP = downsample(AnalogIn__Ch_1_1,r);
T_down = table(T_time,T_Chan_GCamP,T_Chan_Iso,'VariableNames',{'Time','Signal','Control'});

% save csv
fprintf('Save down sampled data to CSV \n')
a = strfind(Dataname,'/B');
filename = Dataname(a+1:end-4);
savename = [savepath,filename,'_doric.csv'];
writetable(T_down,savename,'Delimiter',',','QuoteStrings',true);
toc
end