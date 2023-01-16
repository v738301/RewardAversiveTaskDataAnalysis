clear all
close all

%% main path
datapathDORIC = 'C:\Users\jlab\Documents\DoricData\';
datapathBEH = 'C:\Users\jlab\Documents\BehData\';
datapathTRACK = 'C:\Users\jlab\Documents\MATLAB\DannceProjects\';

%% Load DORIC, BEH and TRACK data into memory
% load neural data from doric recording and save as csv
[file,path,indx] = uigetfile(fullfile(datapathDORIC,'*.csv'));
fullname = fullfile(path,file);
fprintf('Read %s data to memory \n',file)
NeuralStructure = ReadDoric(fullname);

% load beh ttls from cheeta recording and save as csv
AnimalName = file(1:4);
Date = char(datetime(file(9:16),'InputFormat','yyyyMMdd','Format','yyyy-MM-dd'));
SelectTarget = string([Date,'*', AnimalName]);

filename = 'Events.nev';
[datapath_temp] = dir(fullfile(datapathBEH,SelectTarget));
if numel(datapath_temp) > 1
    datapath_temp.name
    prompt = "Which BEH file you want? \n";
    x = input(prompt);
    datapath = fullfile(datapath_temp(x).folder,datapath_temp(x).name);
    names = datapath_temp(x).name;
else
    datapath = fullfile(datapath_temp.folder,datapath_temp.name);
    names = datapath_temp.name;
end
fullname = [datapath,'\',filename];
fprintf('Read %s data to memory \n',names)
BehStrcture = read_nev(fullname);

% extract speed from video
Date = char(datetime(file(9:16),'InputFormat','yyyyMMdd','Format','yyyy_MM_dd'));
SelectTarget = string([AnimalName,'_',Date,'*']);

filename = 'COM\predict_results\com3d.mat';
[datapath_temp] = dir(fullfile(datapathTRACK,SelectTarget));
if numel(datapath_temp) > 1
    datapath_temp.name
    prompt = "Which TRACK file you want? \n";
    x = input(prompt);
    datapath = fullfile(datapath_temp(x).folder,datapath_temp(x).name);
    names = datapath_temp(x).name;
else
    datapath = fullfile(datapath_temp.folder,datapath_temp.name);
    names = datapath_temp.name;
end
fullname = [datapath,'\',filename];
fprintf('Read %s data to memory \n',names)
TrackStrcture = load(fullname);
Com = TrackStrcture.com;
FS_cam = 20;

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

BehTriggerON = DataStrcture.BehTriggerON;
NeuralTriggerON = DataStrcture.NeuralTriggerON;

%% Calculate Delta F over F
ProcessedNeuralStructure = Neural_analysis_Akam_DeltaFOverF(NeuralStructure,0,0);

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
%% denoise com
Com_med = medfilt1(Com,FS_cam/4,'truncate'); %% 1/4 secs

%% interpolate com to align to neural data
Com_align = [];
for i = 1:size(Com_med,2)
    Com_align(:,i) = interp1(NeuralTriggerON(1:length(Com_med),:),Com_med(:,i),NeuralTime,'linear','extrap');
end

%% match the length of com and neural data
lastCom = NeuralTriggerON(length(Com),:);
ComIND = NeuralTime<=lastCom;
Com_align(ComIND==0,:) = repmat(median(Com_align(find(ComIND==0)-FS:find(ComIND==0),:)),sum(ComIND==0),1);

%% smoothing
w = gausswin(length(Com_align)*0.0001, 2.5);
w = w/sum(w);
Com_denoised = filtfilt(w, 1, Com_align);

figure; hold on;
plot(Com_align)
plot(Com_denoised)

%%
figure; plot3(Com_denoised(:,1),Com_denoised(:,2),Com_denoised(:,3))
figure; hold on;
plot(NeuralTime,Com_denoised)
plot(NeuralTime,ComIND*100)

%% Calculate speed from Com
Com_speed = sqrt((Com_denoised(2:end,1)-Com_denoised(1:end-1,1)).^2 + (Com_denoised(2:end,2)-Com_denoised(1:end-1,2)).^2 + (Com_denoised(2:end,3)-Com_denoised(1:end-1,3)).^2);
Com_speed = Com_speed./diff(NeuralTime);
Com_speed = [Com_speed(1,1);Com_speed];
Z_Com_speed = zscore(Com_speed);
%%
Delat_Chan_GCamP_highpass = MotionCorrected_GCamP./Chan_GCamP_Baseline;
Z_Delat_Chan_GCamP_highpass = zscore(Delat_Chan_GCamP_highpass);

figure; hold on;
plot(NeuralTime,Z_Delat_Chan_GCamP_highpass)
plot(NeuralTime,Z_Com_speed)

%% correlate spedd and neural signal
A = Z_Delat_Chan_GCamP_highpass';
B = Z_Com_speed';

figure;
[xcf,lags,~,h] = crosscorr(A,B,NumLags=1200);
lag = lags(argmax(xcf));
% lag = 0;

LagInd1 = (1+lag:length(A));
LagInd2 = (1:length(A)-lag);

A = Z_Delat_Chan_GCamP_highpass(LagInd1,1)'; %% latter
B = zscore(Com_speed(LagInd2,1))'; %% sonner

figure; hold on;
scatter(A(1:1200:end), B(1:1200:end));
[rcof,pval]=corrcoef(A,B);
text(0.2,0.9,['R = ',num2str(round(rcof(1,2),3))],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold');
text(0.2,0.9-0.08,['p < 10^','{',num2str(ceil(log10(pval(2,1)))),'}'],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold')
C=regress(B',[repmat(1,numel(A),1),A']);
plot(linspace(min(A),max(A),11),linspace(min(A),max(A),11).*C(2)+C(1),'--','linewidth',3);
xlabel('Neural')
ylabel('Speed')

%% thresholding speed
A = Z_Delat_Chan_GCamP_highpass';
B = Z_Com_speed';

A(B<0) = [];
B(B<0) = [];

figure; hold on;
scatter(A(1:1200:end), B(1:1200:end));
[rcof,pval]=corrcoef(A,B);
text(0.2,0.9,['R = ',num2str(round(rcof(1,2),3))],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold');
text(0.2,0.9-0.08,['p < 10^','{',num2str(ceil(log10(pval(2,1)))),'}'],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold')
C=regress(B',[repmat(1,numel(A),1),A']);
plot(linspace(min(A),max(A),11),linspace(min(A),max(A),11).*C(2)+C(1),'--','linewidth',3);
xlabel('Neural')
ylabel('Speed')

%% Z-score Delta F/F and speed
Delat_Chan_GCamP_highpass = MotionCorrected_GCamP./Chan_GCamP_Baseline;

figure; hold on;
Z_Delat_Chan_GCamP_highpass = zscore(Delat_Chan_GCamP_highpass);
Z_Com_speed = Z_Com_speed;

plot(NeuralTime,Z_Delat_Chan_GCamP_highpass);
plot(NeuralTime,Z_Com_speed);

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


%% calculate speed around shock
Z_Com_speed = Z_Com_speed;
Ch490 = MotionCorrected_GCamP;
Ch405 = Chan_GCamP_Baseline; % (-inf ~ 0.001)

EventName = {'Shock','Sound1','WP1','WP2','IR1ON','IR2ON'};
EventsON = {};
EventsOFF = {};
names = fieldnames(DataStrcture);
for i = 1:length(names)
    for k = 1:length(EventName)
        if and(~isempty(strfind(names{i},EventName{k})), ~isempty(strfind(names{i},'ON')))
            ONtime = DataStrcture.(names{i});
            OFFtime = DataStrcture.([names{i}(1:end-2),'OFF']);
            EventsON{k} = [ONtime];
            EventsOFF{k} = [OFFtime];
        end
    end
end

NonOverlapping = 1;
SpeacialTime = 0;

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

% Get event TS
EventID = 1;
EventTS = EventsON{EventID};

if SpeacialTime == 1
    [~,b,~] = sparse_distanceXY(EventsON{1},EventsON{2});
    FirstSounds = [EventsON{2}(1),b];
    FirstSounds(isinf(FirstSounds)) = [];
    EventTS = FirstSounds;
    RwardTime1 = EventsON{3}(find(diff([0;EventsON{3}])>30));
    RwardTime2 = EventsON{4}(find(diff([0;EventsON{4}])>30));
    EventTS = sort([RwardTime1';RwardTime2']);
    %     [~,RewLick1] = sparse_distanceXY(RwardTime1,EventsON{5});
    %     [~,RewLick2] = sparse_distanceXY(RwardTime2,EventsON{6});
    %     EventTS = sort([RewLick1';RewLick2']);
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
    [~, CSidx(i,1)]=min(abs(NeuralTime(:,:)-EventTS(i)));
end

% Obtain the DeltaF/F for each event window
CSTS=[];
counter=1;
clear DF_Event DF_F DF_Base DF_ZScore F490CSBL F405CSBL F405 F490 bls Y_Fit ZSpeed SP_ZScore
for i=1:length(CSidx)
    % Check CS events are not earlier than session start and not latter than session end
    if CSidx(i)-(BaselineWind+BaselineWind2)<=0 || CSidx(i)+PostWind > length(NeuralTime)
    else
        % Baseline Signal
        F490CSBL(1,:)=Ch490((CSidx(i)-(BaselineWind+BaselineWind2)):(CSidx(i)-(BaselineWind)));

        % PSTH Signal and baseline
        CSTS=(-PreWind:PostWind)./FS;
        F490(1,:)=Ch490((CSidx(i)-PreWind):(CSidx(i)+PostWind));
        ZSpeed(1,:)=Z_Com_speed((CSidx(i)-PreWind):(CSidx(i)+PostWind));

        DF_Event(:,i)=F490;
        DF_Base(:,i)=F490CSBL;
        DF_ZScore(:,counter)=0.6745.*(DF_Event(:,i)-median(DF_Base(:,i)))./mad(DF_Base(:,i)); %% Normalized by baseline activity
        SP_ZScore(:,counter)=ZSpeed(1,:);

        counter=counter+1;
        clear DF_Event DF_Base F490CSBL F405 F490 bls Y_Fit ZSpeed
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
        tmp3(:,end+1)=median(SP_ZScore(i:end,:),1);
    else
        tmp(1,end+1)=median(CSTS(i:i+bin));
        tmp2(:,end+1)=median(DF_ZScore(i:i+bin,:),1);
        tmp3(:,end+1)=median(SP_ZScore(i:i+bin,:),1);
    end
end

CSTS_bin=tmp; % timestamps
DF_ZScore_bin=tmp2; % PSTH matrix (Z-score)
SP_ZScore_bin=tmp3; % PSTH matrix for speed (Z-score)
CSTrace1=(mean(DF_ZScore_bin,1)); % PSTH matrix (Z-score) mean
CSTrace1SEM = (std(DF_ZScore_bin,1)./sqrt(size(DF_ZScore_bin,1)));
SPTrace1=(mean(SP_ZScore_bin,1)); % PSTH matrix (Z-score) mean for speed
SPTrace1SEM = (std(SP_ZScore_bin,1)./sqrt(size(SP_ZScore_bin,1)));

% Plotting PSTH
% Z-value (subtract baseline)
figure;
subplot(2,1,1); hold on;
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

subplot(2,1,2); hold on;
shadedErrorBar(CSTS_bin,SPTrace1,SPTrace1SEM,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,SPTrace1,'-k','LineWidth',2)
title('Z-score PSTH', 'Fontsize', 18);
xlim([-Pre Post])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(SPTrace1));
CSmin=min(min(SPTrace1));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')
set(gca,'fontsize',18);

figure;
subplot(2,1,1);
imagesc(CSTS_bin,[1:size(DF_ZScore_bin,1)],DF_ZScore_bin,[-3,3]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([-Pre Post])
set(gca,'fontsize',18);

subplot(2,1,2);
imagesc(CSTS_bin,[1:size(SP_ZScore_bin,1)],SP_ZScore_bin,[-3,3]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([-Pre Post])
set(gca,'fontsize',18);

%% Trace with neural activity as color
Delat_Chan_GCamP_highpass = MotionCorrected_GCamP./Chan_GCamP_Baseline;
Z_Delat_Chan_GCamP_highpass = zscore(Delat_Chan_GCamP_highpass);
Z_Com_speed = Z_Com_speed;

figure; scatter3(Com_denoised(:,1),Com_denoised(:,2),Com_denoised(:,3),1,MotionCorrected_GCamP)
figure; scatter3(Com_denoised(:,1),Com_denoised(:,2),Com_denoised(:,3),1,Com_speed)

%%  shock-speed Correlation within shock [0,25]
% Get event TS % EventName = {'Shock','Sound1','WP1','WP2','IR1ON','IR2ON'};
EventID = 1;
EventTS = EventsON{EventID};

CSidx=[];
for i=1:length(EventTS)
    [~, CSidx(i,1)]=min(abs(NeuralTime(:,:)-EventTS(i)));
end
TimeIndex = unique(bsxfun(@plus, CSidx, [0:25*FS]));
OutTimeIndex = setdiff(1:length(NeuralTime),TimeIndex);

% correlate spedd and neural signal around timestamp
A = Z_Delat_Chan_GCamP_highpass(TimeIndex)';
B = zscore(Com_speed(TimeIndex))';

% thresholding speed
A(B<0) = [];
B(B<0) = [];

figure; hold on;
scatter(A(1:1:end), B(1:1:end));
[rcof,pval]=corrcoef(A,B);
text(0.2,0.9,['R = ',num2str(round(rcof(1,2),3))],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold');
text(0.2,0.9-0.08,['p < 10^','{',num2str(ceil(log10(pval(2,1)))),'}'],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold')
C=regress(B',[repmat(1,numel(A),1),A']);
plot(linspace(min(A),max(A),11),linspace(min(A),max(A),11).*C(2)+C(1),'--','linewidth',3);
title("Speed-Neural Correlation around stimulus")
xlabel('Neural')
ylabel('Speed')

%% binning speed and neural signal
% Z_Delat_Chan_GCamP_highpass
bins = [min(Z_Com_speed)-1:1:max(Z_Com_speed)+1];
DisIndSpeed = discretize(Z_Com_speed,bins);
discretedSpeed = bins(DisIndSpeed);

clear AvgSpeed SEMSpeed
for i = 1:length(bins)
    AvgSpeed(i) = mean(Z_Delat_Chan_GCamP_highpass(find(DisIndSpeed==i)));
    SEMSpeed(i) = std(Z_Delat_Chan_GCamP_highpass(find(DisIndSpeed==i)))./sqrt(numel(find(DisIndSpeed==i)));
end

figure; hold on;
shadedErrorBar(bins,AvgSpeed,SEMSpeed,{'-b','color',[0,0,0]},0.5)
plot(bins,AvgSpeed,'-k','LineWidth',2)
title("Binning Speed see Neural")
xlabel('Speed')
ylabel('Neural')

% binning around timestamp
bins = [min(Z_Com_speed)-1:1:max(Z_Com_speed)+1];
DisIndSpeed = discretize(Z_Com_speed(TimeIndex),bins);
discretedSpeed = bins(DisIndSpeed);

clear AvgSpeed SEMSpeed
for i = 1:length(bins)
    AvgSpeed(i) = mean(Z_Delat_Chan_GCamP_highpass(find(DisIndSpeed==i)));
    SEMSpeed(i) = std(Z_Delat_Chan_GCamP_highpass(find(DisIndSpeed==i)))./sqrt(numel(find(DisIndSpeed==i)));
end

figure; hold on;
shadedErrorBar(bins,AvgSpeed,SEMSpeed,{'-b','color',[0,0,0]},0.5)
plot(bins,AvgSpeed,'-k','LineWidth',2)
title("Binning Speed see Neural - around timestamp")
xlabel('Speed')
ylabel('Neural')

% binning outside timestamp
bins = [min(Z_Com_speed)-1:1:max(Z_Com_speed)+1];
DisIndSpeed = discretize(Z_Com_speed(OutTimeIndex),bins);
discretedSpeed = bins(DisIndSpeed);

clear AvgSpeed SEMSpeed
for i = 1:length(bins)
    AvgSpeed(i) = mean(Z_Delat_Chan_GCamP_highpass(find(DisIndSpeed==i)));
    SEMSpeed(i) = std(Z_Delat_Chan_GCamP_highpass(find(DisIndSpeed==i)))./sqrt(numel(find(DisIndSpeed==i)));
end

figure; hold on;
shadedErrorBar(bins,AvgSpeed,SEMSpeed,{'-b','color',[0,0,0]},0.5)
plot(bins,AvgSpeed,'-k','LineWidth',2)
title("Binning Speed see Neural - outside timestamp")
xlabel('Speed')
ylabel('Neural')