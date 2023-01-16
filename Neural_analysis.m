%% load neural data from doric recording and save as csv
datapath = 'C:\Users\jlab\Documents\DoricData\';
filename = 'B636HighShock-FC-20230104.csv';
fullname = [datapath,filename];
NeuralStructure = ReadDoric(fullname);

%% load beh ttls from cheeta recording and save as csv
datapath = 'C:\Users\jlab\Documents\BehData\2023-01-04_14-09-52_B636_highshock\';
filename = 'Events.nev';
fullname = [datapath,filename];
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

%%
NeuralTime = NeuralStructure.Time;
Chan_GCamP = NeuralStructure.Chan_GCamP;
Chan_Iso = NeuralStructure.Chan_Iso;

%% Fill nans
Chan_GCamP = fillmissing(Chan_GCamP,'nearest');
Chan_Iso = fillmissing(Chan_Iso,'nearest');

%% Signal channel
figure;
plot(NeuralTime,Chan_GCamP);
xlim([NeuralTime(1), NeuralTime(end)]);
xlabel('Time (seconds)');
title('Signal Channel Recording');

%% Isobestic channel 
figure;
plot(NeuralTime,Chan_Iso)
xlim([NeuralTime(1), NeuralTime(end)]);
title('Control Recording');
xlabel('Time (seconds)');

%% Both signal and control
figure; hold on;
plot(NeuralTime,Chan_GCamP);
plot(NeuralTime,Chan_Iso);
xlim([NeuralTime(1), NeuralTime(end)]);
xlabel('Time (seconds)');
title('Signal vs Control');
legend({'GCamP','Isobestic'})

%% Denoising and lowpassed
Chan_GCamP_denoised=medfilt1(Chan_GCamP,10,'truncate'); 
Chan_Iso_denoised=medfilt1(Chan_Iso,10,'truncate');

FS = 12000; % Hz
CutOff = 20; % Hz
[b,a] = butter(2, CutOff/(FS/2),'low');
Chan_GCamP_denoised = filtfilt(b,a, Chan_GCamP_denoised);
Chan_Iso_denoised = filtfilt(b,a, Chan_Iso_denoised);

%% Plot High passed, Denoised signal and Raw signal
figure; hold on;
plot(NeuralTime,Chan_GCamP);
plot(NeuralTime,Chan_Iso);
plot(NeuralTime,Chan_GCamP_denoised);
plot(NeuralTime,Chan_Iso_denoised);
xlim([NeuralTime(1), NeuralTime(end)]);
xlabel('Time (seconds)');
title('Signal vs Control');
legend({'GCamP','Isobestic','GCamP-Denoised','Isobestic-Denoised'})

%% Fitted Signal

bls=polyfit(Chan_Iso_denoised(1:end),Chan_GCamP_denoised(1:end),1);
Y_Fit=bls(1).*Chan_Iso_denoised+bls(2);

figure; hold on;
plot(NeuralTime,Chan_GCamP_denoised);
plot(NeuralTime, Y_Fit);
xlim([NeuralTime(1), NeuralTime(end)]);
xlabel('Time (seconds)');
title('Signal vs Fitted Control');
legend({'GCamP','Fitted Signal'})

%% Delta F/F
Delat_Chan_GCamP_denoised = (Chan_GCamP_denoised(:)-Y_Fit(:))./Y_Fit(:);

figure; hold on;
plot(NeuralTime,Delat_Chan_GCamP_denoised.*100)

Peak=max(Delat_Chan_GCamP_denoised);
EventName = {'Shock','Sound1'};

PatchColor = parula(numel(EventName));
names = fieldnames(DataStrcture);
for i = 1:length(names)
    for k = 1:length(EventName)
        if and(~isempty(strfind(names{i},EventName{k})), ~isempty(strfind(names{i},'ON')))
            ONtime = DataStrcture.(names{i});
            OFFtime = DataStrcture.([names{i}(1:end-2),'OFF']);
            Event = [ONtime,OFFtime];
            for t = 1:length(Event)
                x = [Event(t,1) Event(t,1) Event(t,2) Event(t,2)];
                y=[Peak+7+(k-1), Peak+8+(k-1), Peak+8+(k-1), Peak+7+(k-1)];
                p1= patch(x,y,PatchColor(k,:),'FaceAlpha',0.5,'EdgeColor','none');
            end
        end
    end
end

xlim([NeuralTime(1), NeuralTime(end)]);
ylabel('% \Delta F/F');
xlabel('Time (Seconds)');
title('\Delta F/F for Recording ');

%% Z-score Delta F/F
Delat_Chan_GCamP_denoised = (Chan_GCamP_denoised(:)-Y_Fit(:))./Y_Fit(:);

figure; hold on;
Z_Delat_Chan_GCamP_denoised = zscore(Delat_Chan_GCamP_denoised);
plot(NeuralTime,Z_Delat_Chan_GCamP_denoised);

Peak=max(Z_Delat_Chan_GCamP_denoised);
EventName = {'Shock','Sound1'};

PatchColor = parula(numel(EventName));
names = fieldnames(DataStrcture);
for i = 1:length(names)
    for k = 1:length(EventName)
        if and(~isempty(strfind(names{i},EventName{k})), ~isempty(strfind(names{i},'ON')))
            ONtime = DataStrcture.(names{i});
            OFFtime = DataStrcture.([names{i}(1:end-2),'OFF']);
            Event = [ONtime,OFFtime];
            for t = 1:length(Event)
                x = [Event(t,1) Event(t,1) Event(t,2) Event(t,2)];
                y=[Peak+2+(k-1), Peak+3+(k-1), Peak+3+(k-1), Peak+2+(k-1)];
                p1= patch(x,y,PatchColor(k,:),'FaceAlpha',0.5,'EdgeColor','none');
            end
        end
    end
end

xlim([NeuralTime(1), NeuralTime(end)]);
ylabel('Normalized \Delta F/F (z-score)')
xlabel('Time (Seconds)')
title('Normalized \Delta F/F for Recording')

%% PSTH 
NonOverlapping = 1;
SpeacialTime = 0;

% Baselines
Fs = 12000; %12K
BL = 0; %5secs
BL2 = 5;
BaselineWind=round(BL*Fs); 
BaselineWind2=round(BL2*Fs);

% PSTH windows
Pre = 5;
Post = 25;
PreWind=round(Pre*Fs);
PostWind=round(Post*Fs);

bin=1200; %FS/10

Ts = NeuralTime;

% Get event TS
EventID = 1;
EventName = {'Shock','Sound1'};
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
    % special treatment
    [~,b,~] = sparse_distanceXY(EventsON{1},EventsON{2});
    FirstSounds = [EventsON{2}(1),b];
    FirstSounds(isinf(FirstSounds)) = [];
    EventTS = FirstSounds;
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
Ch490 = Chan_GCamP_denoised; 
Ch405 = Chan_Iso_denoised;
counter=1;
clear DF_Event DF_F DF_Base DF_ZScore F490CSBL F405CSBL F405 F490 bls Y_Fit
for i=1:length(CSidx)
    % Check CS events are not earlier than session start and not latter than session end
    if CSidx(i)-(BaselineWind+BaselineWind2)<=0 || CSidx(i)+PostWind > length(Ts)
    else
        % Baseline
        F490CSBL(1,:)=Ch490((CSidx(i)-(BaselineWind+BaselineWind2)):(CSidx(i)-(BaselineWind)));
        F405CSBL(1,:)=Ch405((CSidx(i)-(BaselineWind+BaselineWind2)):(CSidx(i)-(BaselineWind)));

        % PSTH
        CSTS=(-PreWind:PostWind)./Fs;
        F405(1,:)=Ch405((CSidx(i)-PreWind):(CSidx(i)+PostWind));
        F490(1,:)=Ch490((CSidx(i)-PreWind):(CSidx(i)+PostWind));

        % Scale and fit data
        bls=polyfit(F405,F490,1); % PSTH fitting
        Y_Fit=bls(1).*F405+bls(2);

        blsbase=polyfit(F405CSBL,F490CSBL,1); % Baseline fitting
        Y_Fit_base=blsbase(1).*F405CSBL+blsbase(2);

        DF_Event(:,i)=F490-Y_Fit; 
        DF_F(:,i)=DF_Event(:,i)./(Y_Fit'); 
        DF_Base(:,i)=F490CSBL-Y_Fit_base;
        DF_ZScore(:,counter)=(DF_Event(:,i)-median(DF_Base(:,i)))./mad(DF_Base(:,i)); %Z-Score

        counter=counter+1;
        clear DF_Event DF_Base F490CSBL F405CSBL F405 F490 bls Y_Fit
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
imagesc(CSTS_bin,[1:size(DF_ZScore_bin,1)],DF_ZScore_bin); 
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
imagesc(CSTS_bin,[1:size(DF_F_bin,1)],(DF_F_bin)); 
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