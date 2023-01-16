%% Analize Photometry data from rat performing Reward-Shock Test
clear all
close all
%% load neural data from doric recording and save as csv
datapath = 'C:\Users\jlab\Documents\DoricData\';
[file,path,indx] = uigetfile(fullfile(datapath,'*.csv'));
fullname = [datapath,file];
NeuralStructure = ReadDoric(fullname);

%% load beh ttls from cheeta recording and save as csv
filename = 'Events.nev';
[datapath] = uigetdir('C:\Users\jlab\Documents\BehData\');
fullname = [datapath,'\',filename];
BehStrcture = read_nev(fullname);

%% interpolate beh timestamps to neural time space
interptOn = 1;
InterpBehStrcture = Neural2BehTime(BehStrcture,NeuralStructure.T_CamTrigON);
if interptOn == 1
    DataStrcture = InterpBehStrcture;
end
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

%% Calculate Delta F over F
ProcessedNeuralStructure = Neural_analysis_Akam_DeltaFOverF(NeuralStructure,0,1);

FS = ProcessedNeuralStructure.FS;
NeuralTime = ProcessedNeuralStructure.NeuralTime;
Chan_GCamP_denoised = ProcessedNeuralStructure.Chan_GCamP_denoised;
Chan_Iso_denoised = ProcessedNeuralStructure.Chan_Iso_denoised;
Estimated_motion_Bble = ProcessedNeuralStructure.Estimated_motion_Bble;
MotionCorrected_GCamP_Bble = ProcessedNeuralStructure.MotionCorrected_GCamP_Bble;
Chan_GCamP_highpass = ProcessedNeuralStructure.Chan_GCamP_highpass;
Chan_Iso_highpass = ProcessedNeuralStructure.Chan_Iso_highpass;
Estimated_motion = ProcessedNeuralStructure.Estimated_motion;
MotionCorrected_GCamP = ProcessedNeuralStructure.MotionCorrected_GCamP;
Chan_GCamP_Baseline = ProcessedNeuralStructure.Chan_GCamP_Baseline;

% filter high frequency noise
% Chan_GCamP_denoised (-inf ~ 10 Hz)
% Chan_Iso_denoised (-inf ~ 10 Hz)

% Yfit before photobleaching correction
% Estimated_motion_Bble (Yfit ~ a*Chan_Iso_denoised + b)
% MotionCorrected_GCamP_Bble (-inf ~ 10 Hz) minus motion power

% Correct for photobleaching
% Chan_GCamP_highpass (0.1 ~ 10 Hz)
% Chan_Iso_highpass (0.1 ~ 10 Hz)

% correct for motion artifact
% Estimated_motion (Yfit ~ a*Chan_Iso_highpass + b)
% MotionCorrected_GCamP (0.1 ~ 10 Hz) minus motion power

% Low frequency base line (contain photobleacing trend)
% Chan_GCamP_Baseline (-inf ~ 0.001)

%%
figure; hold on;
plot(MotionCorrected_GCamP_Bble)
plot(MotionCorrected_GCamP)
legend({'MotionCorrected_GCamP_Bble','MotionCorrected_GCamP'})

%% Delta F/F
Delat_Chan_GCamP_highpass = MotionCorrected_GCamP./Chan_GCamP_Baseline;

figure; hold on;
plot(NeuralTime,Delat_Chan_GCamP_highpass.*100)

Peak=max(Delat_Chan_GCamP_highpass);
EventName = {'Shock','Sound1','IR1','IR2','WP1','WP2'};

PatchColor = parula(numel(EventName));
names = fieldnames(DataStrcture);
for i = 1:length(names)
    for k = 1:length(EventName)
        if and(~isempty(strfind(names{i},EventName{k})), ~isempty(strfind(names{i},'ON')))
            ONtime = DataStrcture.(names{i});
            OFFtime = DataStrcture.([names{i}(1:end-2),'OFF']);
            Event = [ONtime,OFFtime];
            if ~isempty(Event)
                for t = 1:size(Event,1)
                    x = [Event(t,1) Event(t,1) Event(t,2) Event(t,2)];
                    y=[Peak+7+(k-1), Peak+8+(k-1), Peak+8+(k-1), Peak+7+(k-1)];
                    p1= patch(x,y,PatchColor(k,:),'FaceAlpha',0.5,'EdgeColor','none');
                end
            end
        end
    end
end

xlim([NeuralTime(1), NeuralTime(end)]);
ylabel('% \Delta F/F');
xlabel('Time (Seconds)');
title('\Delta F/F for Recording ');

%% Z-score Delta F/F
Delat_Chan_GCamP_highpass = MotionCorrected_GCamP./Chan_GCamP_Baseline;

figure; hold on;
Z_Delat_Chan_GCamP_highpass = zscore(Delat_Chan_GCamP_highpass);
plot(NeuralTime,Z_Delat_Chan_GCamP_highpass);

Peak=max(Z_Delat_Chan_GCamP_highpass);
EventName = {'Shock','Sound1','IR1','IR2','WP1','WP2'};

PatchColor = parula(numel(EventName));
names = fieldnames(DataStrcture);
for i = 1:length(names)
    for k = 1:length(EventName)
        if and(~isempty(strfind(names{i},EventName{k})), ~isempty(strfind(names{i},'ON')))
            ONtime = DataStrcture.(names{i});
            OFFtime = DataStrcture.([names{i}(1:end-2),'OFF']);
            Event = [ONtime,OFFtime];
            if ~isempty(Event)
                for t = 1:size(Event,1)
                    x = [Event(t,1) Event(t,1) Event(t,2) Event(t,2)];
                    y=[Peak+7+(k-1), Peak+8+(k-1), Peak+8+(k-1), Peak+7+(k-1)];
                    p1= patch(x,y,PatchColor(k,:),'FaceAlpha',0.5,'EdgeColor','none');
                end
            end
        end
    end
end

xlim([NeuralTime(1), NeuralTime(end)]);
ylabel('Normalized \Delta F/F (z-score)')
xlabel('Time (Seconds)')
title('Normalized \Delta F/F for Recording')

%% Plot PSTH

Ch490 = MotionCorrected_GCamP;
Ch405 = Chan_GCamP_Baseline; % (-inf ~ 0.001)

% Ch490 = MotionCorrected_GCamP_Bble;
% Ch405 = Chan_GCamP_Baseline; % (-inf ~ 0.001)

NonOverlapping = 1;
SpeacialTime = 0;

% sound strat from 0.75 secs !!

% Baselines in secs
BL = 20;
BL2 = 5; % Baselines = (CS_Time-(BaselineWind+BaselineWind2)):(CS_Time-(BaselineWind))
BaselineWind=round(BL*FS);
BaselineWind2=round(BL2*FS);

% PSTH windows in secs
Pre = 5;
Post = 25;
PreWind=round(Pre*FS);
PostWind=round(Post*FS);

bin=FS/5; %FS/10

Ts = NeuralTime;

% Get event TS
EventID = 1;
EventName = {'Shock','Sound1','WP1','WP2','IR1ON','IR2ON'};
EventsON = {};
names = fieldnames(DataStrcture);
for i = 1:length(names)
    for k = 1:length(EventName)
        if and(~isempty(strfind(names{i},EventName{k})), ~isempty(strfind(names{i},'ON')))
            ONtime = DataStrcture.(names{i});
            OFFtime = DataStrcture.([names{i}(1:end-2),'OFF']);
            EventsON{k} = [ONtime];
        end
    end
end

EventTS = EventsON{EventID};

if SpeacialTime == 1
    [~,b,~] = sparse_distanceXY(EventsON{1},EventsON{2});
    FirstSounds = [EventsON{2}(1),b];
    FirstSounds(isinf(FirstSounds)) = [];
    EventTS = FirstSounds;
    %     RwardTime1 = EventsON{3}(find(diff([0;EventsON{3}])>30));
    %     [~,RewLick1] = sparse_distanceXY(RwardTime1,EventsON{5});
    %     RwardTime2 = EventsON{4}(find(diff([0;EventsON{4}])>30));
    %     [~,RewLick2] = sparse_distanceXY(RwardTime2,EventsON{6});
    %     EventTS = sort([RewLick1';RewLick2']);
    %     EventTS = sort([RwardTime1';RwardTime2']);
end

% Remove overlapping timestamps
if NonOverlapping == 1
    tmp=[];
    CurrEvent=EventTS(1);
    for i=1:length(EventTS)-1
        if EventTS(i+1)-CurrEvent>Pre+Post
            tmp(end+1,1)=CurrEvent;
            CurrEvent=EventTS(i+1);
        else
        end
    end
    tmp(end+1,1)=EventTS(length(EventTS));
    EventTS=tmp;
end

CSidx=[];
for i=1:length(EventTS)
    [~, CSidx(i,1)]=min(abs(Ts(:,:)-EventTS(i)));
end

% Obtain the DeltaF/F for each event window
CSTS=[];
counter=1;
clear DF_Event DF_F DF_Base DF_ZScore F490CSBL F405CSBL F405 F490 bls Y_Fit
for i=1:length(CSidx)
    % Check CS events are not earlier than session start and not latter than session end
    if CSidx(i)-(BaselineWind+BaselineWind2)<=0 || CSidx(i)+PostWind > length(Ts)
    else
        % Baseline Signal
        F490CSBL(1,:)=Ch490((CSidx(i)-(BaselineWind+BaselineWind2)):(CSidx(i)-(BaselineWind)));

        % PSTH Signal and baseline
        CSTS=(-PreWind:PostWind)./FS;
        F405(1,:)=Ch405((CSidx(i)-PreWind):(CSidx(i)+PostWind));
        F490(1,:)=Ch490((CSidx(i)-PreWind):(CSidx(i)+PostWind));

        DF_Event(:,i)=F490;
        DF_F(:,i)=DF_Event(:,i)./F405'; %% Normalized by itself
        DF_Base(:,i)=F490CSBL;
        DF_ZScore(:,counter)=0.6745.*(DF_Event(:,i)-median(DF_Base(:,i)))./mad(DF_Base(:,i)); %% Normalized by baseline activity

        counter=counter+1;
        clear DF_Event DF_Base F490CSBL F405 F490 bls Y_Fit
    end
end

% Binning PSTH matrix (Baseline subtracted and not)
tmp=[];
tmp2=[];
tmp3=[];
for i=1:bin:length(CSTS)
    if i+bin>length(CSTS)
        tmp(1,end+1)=median(CSTS(i:end));
        tmp2(:,end+1)=median(DF_ZScore(i:end,:),1);
        tmp3(:,end+1)=median(DF_F(i:end,:),1);
    else
        tmp(1,end+1)=median(CSTS(i:i+bin));
        tmp2(:,end+1)=median(DF_ZScore(i:i+bin,:),1);
        tmp3(:,end+1)=median(DF_F(i:i+bin,:),1);
    end
end

CSTS_bin=tmp; % timestamps

DF_ZScore_bin=tmp2; % PSTH matrix (Z-score)
CSTrace1=(mean(DF_ZScore_bin,1)); % PSTH matrix (Z-score) mean
CSTrace1SEM = (std(DF_ZScore_bin,1)./sqrt(size(DF_ZScore_bin,1)));

DF_F_bin=tmp3*100; % PSTH matrix (float value), *100 to change into percentage
CSTrace2=(mean(DF_F_bin,1)); % PSTH matrix (float value) mean
CSTrace2SEM = (std(DF_F_bin,1)./sqrt(size(DF_F_bin,1)));

% Plotting PSTH
% Z-value (subtract baseline)
figure;
subplot(2,1,1);
imagesc(CSTS_bin,[1:size(DF_ZScore_bin,1)],DF_ZScore_bin,[-3,3]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([-Pre Post])
set(gca,'fontsize',18);

subplot(2,1,2); hold on;
shadedErrorBar(CSTS_bin,CSTrace1,CSTrace1SEM,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,CSTrace1,'-k','LineWidth',2)
title('Z-score PSTH', 'Fontsize', 18);
xlim([-Pre Post])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(CSTrace1));
CSmin=min(min(CSTrace1));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')

set(gca,'fontsize',18);

% Delta F/F
figure;
subplot(2,1,1);
imagesc(CSTS_bin,[1:size(DF_F_bin,1)],(DF_F_bin),[-0.5,0.5]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' \Delta F/F PSTH'],'fontsize', 18)
xlim([-Pre Post])
set(gca,'fontsize',18);

subplot(2,1,2); hold on;
shadedErrorBar(CSTS_bin,CSTrace2,CSTrace2SEM,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,CSTrace2,'-k','LineWidth',2)
title('Trial Based \Delta F/F', 'Fontsize', 18);
xlim([-Pre Post])
xlabel('Time (s)', 'FontSize',18)
ylabel('% \Delta F/F','fontsize', 18);
CSmax=max(max(CSTrace2));
CSmin=min(min(CSTrace2));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')

set(gca,'fontsize',18);
