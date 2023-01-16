%% load neural data from doric recording and save as csv
datapath = '/Users/hsiehkunlin/Desktop/Data/Doric/';
filename = 'B636-20221230-FC_0000.csv';
fullname = [datapath,filename];
[T_CamTrigON, T_CamTrigOFF] = Save2Doric(fullname,datapath);

%% load beh ttls from cheeta recording and save as csv
datapath = '/Users/hsiehkunlin/Desktop/Data/BEH/2022-12-30_16-01-46_B636/';
filename = 'Events.nev';
fullname = [datapath,filename];
DataStrcture = read_nev(fullname);
Save2Event(DataStrcture,datapath,T_CamTrigON)

%% remove artifacts
% datax(:,x) = hampel(datax(:,x));
%%
Cam1ON = DataStrcture.Cam1ON;
Cam2ON = DataStrcture.Cam2ON;
Cam3ON = DataStrcture.Cam3ON;
CamTrigON = DataStrcture.CamTrigON;
IR1ON = DataStrcture.IR1ON;
IR2ON = DataStrcture.IR2ON;
ShockON = DataStrcture.ShockON;
Sound1ON = DataStrcture.Sound1ON;
WP1ON =  DataStrcture.WP1ON;
WP2ON = DataStrcture.WP2ON;

Cam1OFF = DataStrcture.Cam1OFF;
Cam2OFF = DataStrcture.Cam2OFF;
Cam3OFF = DataStrcture.Cam3OFF;
CamTrigOFF = DataStrcture.CamTrigOFF;
IR1OFF = DataStrcture.IR1OFF;
IR2OFF = DataStrcture.IR2OFF;
ShockOFF = DataStrcture.ShockOFF;
Sound1OFF = DataStrcture.Sound1OFF;
WP1OFF =  DataStrcture.WP1OFF;
WP2OFF = DataStrcture.WP2OFF;

%%
% NeuralTriggerON = T_CamTrigON;
% BehTriggerON = DataStrcture.CamTrigON;
% BehTriggerON = BehTriggerON(1:min(numel(BehTriggerON),numel(NeuralTriggerON)));
% NeuralTriggerON = NeuralTriggerON(1:min(numel(BehTriggerON),numel(NeuralTriggerON)));
% 
% Cam1ON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Cam1ON,'linear','extrap');
% Cam2ON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Cam2ON,'linear','extrap');
% Cam3ON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Cam3ON,'linear','extrap');
% CamTrigON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.CamTrigON,'linear','extrap');
% IR1ON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.IR1ON,'linear','extrap');
% IR2ON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.IR2ON,'linear','extrap');
% ShockON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.ShockON,'linear','extrap');
% Sound1ON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Sound1ON,'linear','extrap');
% WP1ON =  interp1(BehTriggerON,NeuralTriggerON,DataStrcture.WP1ON,'linear','extrap');
% WP2ON = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.WP2ON,'linear','extrap');
% 
% Cam1OFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Cam1OFF,'linear','extrap');
% Cam2OFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Cam2OFF,'linear','extrap');
% Cam3OFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Cam3OFF,'linear','extrap');
% CamTrigOFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.CamTrigOFF,'linear','extrap');
% IR1OFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.IR1OFF,'linear','extrap');
% IR2OFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.IR2OFF,'linear','extrap');
% ShockOFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.ShockOFF,'linear','extrap');
% Sound1OFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.Sound1OFF,'linear','extrap');
% WP1OFF =  interp1(BehTriggerON,NeuralTriggerON,DataStrcture.WP1OFF,'linear','extrap');
% WP2OFF = interp1(BehTriggerON,NeuralTriggerON,DataStrcture.WP2OFF,'linear','extrap');

%%
pmat
%%
ind = 1:1000;
figure; hold on;
% plot(Cam1ON(ind),1,['ro'])
% plot(Cam2ON(ind),2,['go'])
% plot(Cam3ON(ind),3,['bo'])
% plot(CamTrigON(ind),4,['ko'])

plot(IR1ON(:),5,['ro'])
plot(WP1ON(:),6,['bo'])
plot(IR2ON(:),7,['go'])
plot(WP2ON(:),8,['ko'])

plot(ShockON(:),9,['co'])
plot(Sound1ON(:),10,['go'])
% 
% plot(Cam1OFF(ind),1,['r^'])
% plot(Cam2OFF(ind),2,['g^'])
% plot(Cam3OFF(ind),3,['b^'])
% plot(CamTrigOFF(ind),4,['k^'])

plot(IR1OFF(:),5,['r^'])
plot(WP1OFF(:),6,['b^'])
plot(IR2OFF(:),7,['g^'])
plot(WP2OFF(:),8,['k^'])

plot(ShockOFF(:),9,['c^'])
plot(Sound1OFF(:),10,['g^'])

yticks([1:10])
yticklabels({'Cam1','Cam2','Cam3','CamTrig','IR1','WP1','IR2','WP2','Shock','Sound1'})

ylim([0,11])
%%
figure; plot(IR1OFF(1:length(IR1ON)) - IR1ON)
%%
figure; hold on;
plot(diff(Cam1ON))
plot(diff(Cam2ON))
plot(diff(Cam3ON))
plot(diff(CamTrigON))
%%
figure; hold on;
plot(Cam1ON(1:1000),0,'ko')
plot(Cam2ON(1:1000),2,'ro')
plot(Cam3ON(1:1000),4,'go')
plot(CamTrigON(1:1000),6,'bo')
ylim([-5,10])