clear all
close all

%% main path
ProjectName = "UPS";
if ProjectName == "FC"
    datapathDORIC = "D:\CnF_photometry\FC\Doric\";
    datapathBEH = "D:\CnF_photometry\FC\BEH\";
    datapathTRACK = "C:\Users\jlab\Documents\MATLAB\DannceProjects\";
    datapathMAT = "D:\CnF_photometry\FC\ALLmats";
    figurepath = "D:\CnF_photometry\FC\Figures";
    AllPath = [datapathDORIC,datapathBEH,datapathTRACK];
elseif ProjectName == "UPS"
    datapathDORIC = "D:\CnF_photometry\UnpredictedShock\Doric\";
    datapathBEH = "D:\CnF_photometry\UnpredictedShock\BEH\";
    datapathTRACK = "C:\Users\jlab\Documents\MATLAB\DannceProjects\";
    datapathMAT = "D:\CnF_photometry\UnpredictedShock\ALLmats";
    figurepath = "D:\CnF_photometry\UnpredictedShock\Figures";
    AllPath = [datapathDORIC,datapathBEH,datapathTRACK];
else
end

%% Load DORIC, BEH and TRACK data into memory
% read doric data from folder
% [file] = uigetfile(fullfile(datapathDORIC,'*.csv'));
% AllStructure = Read3structure(file,AllPath);
% NeuralStructure = AllStructure.NeuralStructure;
% BehStrcture = AllStructure.BehStrcture;
% TrackStrcture = AllStructure.TrackStrcture;
% Com = TrackStrcture.com;
% FS_cam = 20;

%% Or read data from mats
% [file,path] = uigetfile(fullfile(datapathMAT,'*.mat'));
% AllStructure = load(fullfile(path,file));

files = dir(fullfile(datapathMAT,'B*.mat'));
Stimuli = "Shock";
if ProjectName == "FC"
    if Stimuli == "Shock"
        SessionsID = [1,4:6,10:length(files)];
    elseif Stimuli == "Reward"
        SessionsID = [1:3,5,7,8,10:length(files)];
    end
elseif ProjectName == "UPS"
    Stimuli = "Shock";
    SessionsID = [1:length(files)];
end
for session = SessionsID
    session
    path = files(session).folder; file = files(session).name;
    AllStructure = load(fullfile(path,file));

    AllStructure = AllStructure.AllStructure;
    NeuralStructure = AllStructure.NeuralStructure;
    BehStrcture = AllStructure.BehStrcture;
    TrackStrcture = AllStructure.TrackStrcture;
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
    %% Get signal from structure
    FS = ProcessedNeuralStructure.FS;
    NeuralTime = ProcessedNeuralStructure.NeuralTime;
    %     Chan_GCamP_denoised = ProcessedNeuralStructure.Chan_GCamP_denoised;
    %     Chan_Iso_denoised = ProcessedNeuralStructure.Chan_Iso_denoised;
    %     Estimated_motion_Bble = ProcessedNeuralStructure.Estimated_motion_Bble;
    %     MotionCorrected_GCamP_Bble = ProcessedNeuralStructure.MotionCorrected_GCamP_Bble;
    %     Chan_GCamP_highpass = ProcessedNeuralStructure.Chan_GCamP_highpass;
    %     Chan_Iso_highpass = ProcessedNeuralStructure.Chan_Iso_highpass;
    %     Estimated_motion = ProcessedNeuralStructure.Estimated_motion;
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

    Delat_Chan_GCamP_highpass_HSR = MotionCorrected_GCamP./Chan_GCamP_Baseline;

    %% denoise com
    Com_med = medfilt1(Com,FS_cam/4,'truncate'); %% 1/4 secs

    %% interpolate com to align to neural data then down sampling both
    AlignToNeuralOrBEH = "BEH";
    if AlignToNeuralOrBEH == "Neural"
        Com_align = [];
        for i = 1:size(Com_med,2)
            Com_align(:,i) = interp1(NeuralTriggerON(1:length(Com_med),:),Com_med(:,i),NeuralTime,'linear',median(Com_med));
        end
        %% match the length of com and neural data
        lastCom = NeuralTriggerON(length(Com),:);
        ComIND = NeuralTime<=lastCom;
        Com_align(ComIND==0,:) = repmat(median(Com_align),sum(ComIND==0),1);
    elseif AlignToNeuralOrBEH == "BEH"
        FirstNeu = NeuralTime(1);
        Com_align = Com_med(1:min(length(NeuralTriggerON),length(Com_med)),:);
        Delat_Chan_GCamP_highpass = interp1(NeuralTime,Delat_Chan_GCamP_highpass_HSR,NeuralTriggerON(1:min(length(NeuralTriggerON),length(Com_med)),:),'linear',median(Delat_Chan_GCamP_highpass_HSR));
        MotionCorrected_GCamP = interp1(NeuralTime,MotionCorrected_GCamP,NeuralTriggerON(1:min(length(NeuralTriggerON),length(Com_med)),:),'linear',median(Delat_Chan_GCamP_highpass_HSR));
        NeuralTime = NeuralTriggerON(1:min(length(NeuralTriggerON),length(Com_med)),:);
        FS = FS_cam;
        clear Delat_Chan_GCamP_highpass_HSR Chan_GCamP_Baseline
        %% match the length of com and neural data
        NeuIND = NeuralTime>=FirstNeu;
        Delat_Chan_GCamP_highpass(NeuIND==0,:) = repmat(median(Delat_Chan_GCamP_highpass),sum(NeuIND==0),1);
    else
    end

    %% smoothing
    w = gausswin(length(Com_align)*0.0001, 2.5);
    w = w/sum(w);
    Com_denoised = filtfilt(w, 1, Com_align);

    %% Calculate speed from Com
    Com_speed = sqrt((Com_denoised(2:end,1)-Com_denoised(1:end-1,1)).^2 + (Com_denoised(2:end,2)-Com_denoised(1:end-1,2)).^2 + (Com_denoised(2:end,3)-Com_denoised(1:end-1,3)).^2);
    Com_speed = Com_speed./diff(NeuralTime);
    Com_speed = [Com_speed(1,1);Com_speed];
    Z_Com_speed = zscore(Com_speed);

    %% calculate speed around shock
    Z_Com_speed = Z_Com_speed;
    Ch490 = MotionCorrected_GCamP;

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

    if Stimuli == "Shock"
        NonOverlapping = 1;
        SpeacialTime = 0;
        % Baselines in secs
        BL = 20;
    elseif Stimuli == "Reward"
        NonOverlapping = 1;
        SpeacialTime = 1;
        % Baselines in secs
        BL = 0;
    end

    % Baselines in secs
    BL2 = 5; % Baselines = (CS_Time-(BaselineWind+BaselineWind2)):(CS_Time-(BaselineWind))
    BaselineWind=round(BL*FS);
    BaselineWind2=round(BL2*FS);

    % PSTH windows in secs
    Pre = 25;
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
        EventTS = sort([RwardTime1;RwardTime2]);
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

    CSTS_bin{session} = tmp; % timestamps
    DF_ZScore_bin{session} = tmp2; % PSTH matrix (Z-score)
    SP_ZScore_bin{session} = tmp3; % PSTH matrix for speed (Z-score)
    CSTrace1{session} = (mean(DF_ZScore_bin{session},1)); % PSTH matrix (Z-score) mean
    CSTrace1SEM{session} = (std(DF_ZScore_bin{session},1)./sqrt(size(DF_ZScore_bin{session},1)));
    SPTrace1{session} = (mean(SP_ZScore_bin{session},1)); % PSTH matrix (Z-score) mean for speed
    SPTrace1SEM{session} = (std(SP_ZScore_bin{session},1)./sqrt(size(SP_ZScore_bin{session},1)));
    META{session}.trialNum = size(tmp2);
    META{session}.Name = file;
end

% Save mat
ALLTimeStamp.CSTS_bin = CSTS_bin;
ALLTimeStamp.DF_ZScore_bin = DF_ZScore_bin;
ALLTimeStamp.SP_ZScore_bin = SP_ZScore_bin;
ALLTimeStamp.CSTrace1 = CSTrace1;
ALLTimeStamp.CSTrace1SEM = CSTrace1SEM;
ALLTimeStamp.SPTrace1 = SPTrace1;
ALLTimeStamp.SPTrace1SEM = SPTrace1SEM;
ALLTimeStamp.Meta = META;

if Stimuli == "Shock"
    save(fullfile(datapathMAT,"ALLShock"), "ALLTimeStamp");
elseif Stimuli == "Reward"
    save(fullfile(datapathMAT,"ALLReward"), "ALLTimeStamp");
end

%%
ALLTimeStamp = load(fullfile(datapathMAT,"ALLShock"), "ALLTimeStamp");
ALLTimeStamp = ALLTimeStamp.ALLTimeStamp;
ALLTimeStamp2 = load(fullfile(datapathMAT,"ALLReward"), "ALLTimeStamp");
ALLTimeStamp2 = ALLTimeStamp2.ALLTimeStamp;


CSTS_bin = ALLTimeStamp.CSTS_bin{1};
for i = 1:length(ALLTimeStamp.CSTS_bin)
    try 
        DF_ZScore_bin = [];
        SP_ZScore_bin = [];
        TrialsVector = [];

        DF_ZScore_bin = ALLTimeStamp.DF_ZScore_bin{i};
        SP_ZScore_bin = ALLTimeStamp.SP_ZScore_bin{i};
        TrialsVector = vertcat(TrialsVector,ALLTimeStamp.Meta{i}.trialNum);
    
        CSTrace1=(mean(DF_ZScore_bin,1)); % PSTH matrix (Z-score) mean
        SPTrace1=(mean(SP_ZScore_bin,1)); % PSTH matrix (Z-score) mean for speed
        CSTrace1SEM = (std(DF_ZScore_bin,1)./sqrt(size(DF_ZScore_bin,1)));
        SPTrace1SEM = (std(SP_ZScore_bin,1)./sqrt(size(SP_ZScore_bin,1)));
    
        % Plotting PSTH
        % Z-value (subtract baseline)
        figure("position",[100, 0, 2000, 600]);
        subplot(2,4,1); hold on;
        shadedErrorBar(CSTS_bin,CSTrace1,CSTrace1SEM,{'-b','color',[0,0,0]},0.5)
        plot(CSTS_bin,CSTrace1,'-k','LineWidth',2)
        title(ALLTimeStamp.Meta{i}.Name, 'Fontsize', 18);
        xlim([CSTS_bin(1) CSTS_bin(end)])
        xlabel('Time (s)', 'FontSize',18)
        ylabel('Z-score','fontsize', 18);
        CSmax=max(max(CSTrace1));
        CSmin=min(min(CSTrace1));
        ylim([CSmin CSmax*1.25]);
        plot([0,0],[CSmin,CSmax*1.25],':r')
        set(gca,'fontsize',18);
    
        subplot(2,4,2); hold on;
        shadedErrorBar(CSTS_bin,SPTrace1,SPTrace1SEM,{'-b','color',[0,0,0]},0.5)
        plot(CSTS_bin,SPTrace1,'-k','LineWidth',2)
        title('Z-score PSTH', 'Fontsize', 18);
        xlim([CSTS_bin(1) CSTS_bin(end)])
        xlabel('Time (s)', 'FontSize',18)
        ylabel('Z-score','fontsize', 18);
        CSmax=max(max(SPTrace1));
        CSmin=min(min(SPTrace1));
        ylim([CSmin CSmax*1.25]);
        plot([0,0],[CSmin,CSmax*1.25],':r')
        set(gca,'fontsize',18);
    
        subplot(2,4,3); hold on;
        yline(cumsum(TrialsVector(:,1))+0.5,'LineWidth',1.5)
        imagesc(CSTS_bin,[1:size(DF_ZScore_bin,1)],DF_ZScore_bin,[-2,2]);
        ylabel('Trial #','fontsize', 18)
        title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
        xlim([CSTS_bin(1) CSTS_bin(end)])
        ylim([0.5,sum(TrialsVector(:,1)+0.5)])
        set(gca,'fontsize',18);
    
        subplot(2,4,4); hold on;
        yline(cumsum(TrialsVector(:,1))+0.5,'LineWidth',1.5)
        imagesc(CSTS_bin,[1:size(SP_ZScore_bin,1)],SP_ZScore_bin,[-3,3]);
        ylabel('Trial #','fontsize', 18)
        title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
        xlim([CSTS_bin(1) CSTS_bin(end)])
        ylim([0.5,sum(TrialsVector(:,1)+0.5)])
        set(gca,'fontsize',18);

        DF_ZScore_bin = [];
        SP_ZScore_bin = [];
        TrialsVector = [];
    
        DF_ZScore_bin = ALLTimeStamp2.DF_ZScore_bin{i};
        SP_ZScore_bin = ALLTimeStamp2.SP_ZScore_bin{i};
        TrialsVector = vertcat(TrialsVector,ALLTimeStamp2.Meta{i}.trialNum);
    
        CSTrace1=(mean(DF_ZScore_bin,1)); % PSTH matrix (Z-score) mean
        SPTrace1=(mean(SP_ZScore_bin,1)); % PSTH matrix (Z-score) mean for speed
        if size(DF_ZScore_bin,1) == 1
            CSTrace1SEM = zeros(size(DF_ZScore_bin));
            SPTrace1SEM = zeros(size(SPTrace1SEM));
        else
            CSTrace1SEM = (std(DF_ZScore_bin,1)./sqrt(size(DF_ZScore_bin,1)));
            SPTrace1SEM = (std(SP_ZScore_bin,1)./sqrt(size(SP_ZScore_bin,1)));
        end

        subplot(2,4,5); hold on;
        shadedErrorBar(CSTS_bin,CSTrace1,CSTrace1SEM,{'-b','color',[0,0,0]},0.5)
        plot(CSTS_bin,CSTrace1,'-k','LineWidth',2)
        title('Z-score PSTH', 'Fontsize', 18);
        xlim([CSTS_bin(1) CSTS_bin(end)])
        xlabel('Time (s)', 'FontSize',18)
        ylabel('Z-score','fontsize', 18);
        CSmax=max(max(CSTrace1));
        CSmin=min(min(CSTrace1));
        ylim([CSmin CSmax*1.25]);
        plot([0,0],[CSmin,CSmax*1.25],':r')
        set(gca,'fontsize',18);
    
        subplot(2,4,6); hold on;
        shadedErrorBar(CSTS_bin,SPTrace1,SPTrace1SEM,{'-b','color',[0,0,0]},0.5)
        plot(CSTS_bin,SPTrace1,'-k','LineWidth',2)
        title('Z-score PSTH', 'Fontsize', 18);
        xlim([CSTS_bin(1) CSTS_bin(end)])
        xlabel('Time (s)', 'FontSize',18)
        ylabel('Z-score','fontsize', 18);
        CSmax=max(max(SPTrace1));
        CSmin=min(min(SPTrace1));
        ylim([CSmin CSmax*1.25]);
        plot([0,0],[CSmin,CSmax*1.25],':r')
        set(gca,'fontsize',18);
    
        subplot(2,4,7); hold on;
        yline(cumsum(TrialsVector(:,1))+0.5,'LineWidth',1.5)
        imagesc(CSTS_bin,[1:size(DF_ZScore_bin,1)],DF_ZScore_bin,[-2,2]);
        ylabel('Trial #','fontsize', 18)
        title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
        xlim([CSTS_bin(1) CSTS_bin(end)])
        ylim([0.5,sum(TrialsVector(:,1)+0.5)])
        set(gca,'fontsize',18);
                            
        subplot(2,4,8); hold on;
        yline(cumsum(TrialsVector(:,1))+0.5,'LineWidth',1.5)
        imagesc(CSTS_bin,[1:size(SP_ZScore_bin,1)],SP_ZScore_bin,[-3,3]);
        ylabel('Trial #','fontsize', 18)
        title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
        xlim([CSTS_bin(1) CSTS_bin(end)])
        ylim([0.5,sum(TrialsVector(:,1)+0.5)])
        set(gca,'fontsize',18);
    catch
    end
end

%% save all figure
FolderName = figurepath;   % Your destination folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
sessionName = file(1:end-4);
for iFig = 1:length(FigList)
    FigHandle = FigList(iFig);
    saveas(FigHandle, fullfile(FolderName, strcat(num2str(iFig,'%02.f'),"_","RewardAndShock",'.jpg')));
end
%% Pool all session
load(fullfile(datapathMAT,"ALLShock"), "ALLTimeStamp")
load(fullfile(datapathMAT,"ALLReward"), "ALLTimeStamp")

%%
% for UPS
UPSIND = [1,3,4,7];
PSIND = [2,5,6,8];

CSTS_bin = ALLTimeStamp.CSTS_bin{1};
DF_ZScore_bin = [];
SP_ZScore_bin = [];
TrialsVector = [];
for i = 1:length(ALLTimeStamp.CSTS_bin)
    if ~any(i==[])
        try
            if ProjectName == "FC"
                DF_ZScore_bin = vertcat(DF_ZScore_bin,ALLTimeStamp.DF_ZScore_bin{i});
                SP_ZScore_bin = vertcat(SP_ZScore_bin,ALLTimeStamp.SP_ZScore_bin{i});
                TrialsVector = vertcat(TrialsVector,ALLTimeStamp.Meta{i}.trialNum);
            elseif ProjectName == "UPS"
                DF_ZScore_bin = vertcat(DF_ZScore_bin,ALLTimeStamp.DF_ZScore_bin{i}([UPSIND,PSIND],:));
                SP_ZScore_bin = vertcat(SP_ZScore_bin,ALLTimeStamp.SP_ZScore_bin{i}([UPSIND,PSIND],:));
                TrialsVector = vertcat(TrialsVector,ALLTimeStamp.Meta{i}.trialNum);
            end
        catch
        end
    end
end
CSTrace1=(mean(DF_ZScore_bin,1)); % PSTH matrix (Z-score) mean
SPTrace1=(mean(SP_ZScore_bin,1)); % PSTH matrix (Z-score) mean for speed
CSTrace1SEM = (std(DF_ZScore_bin,1)./sqrt(size(DF_ZScore_bin,1)));
SPTrace1SEM = (std(SP_ZScore_bin,1)./sqrt(size(SP_ZScore_bin,1)));

%%
% Z-value (subtract baseline)
fig = figure("position",[100, 0, 2000, 600]);
subplot(2,2,1); hold on;
shadedErrorBar(CSTS_bin,CSTrace1,CSTrace1SEM,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,CSTrace1,'-k','LineWidth',2)
title('AllAessionCombined', 'Fontsize', 18);
xlim([CSTS_bin(1) CSTS_bin(end)])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(CSTrace1));
CSmin=min(min(CSTrace1));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')
set(gca,'fontsize',18);

subplot(2,2,3); hold on;
shadedErrorBar(CSTS_bin,SPTrace1,SPTrace1SEM,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,SPTrace1,'-k','LineWidth',2)
title('Z-score PSTH', 'Fontsize', 18);
xlim([CSTS_bin(1) CSTS_bin(end)])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(SPTrace1));
CSmin=min(min(SPTrace1));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')
set(gca,'fontsize',18);

subplot(2,2,2); hold on;
yline(cumsum(TrialsVector(:,1))+0.5,'LineWidth',1.5)
imagesc(CSTS_bin,[1:size(DF_ZScore_bin,1)],DF_ZScore_bin,[-2,2]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([CSTS_bin(1) CSTS_bin(end)])
ylim([0.5,sum(TrialsVector(:,1))+0.5])
set(gca,'fontsize',18);

subplot(2,2,4); hold on;
yline(cumsum(TrialsVector(:,1))+0.5,'LineWidth',1.5)
imagesc(CSTS_bin,[1:size(SP_ZScore_bin,1)],SP_ZScore_bin,[-1,6]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([CSTS_bin(1) CSTS_bin(end)])
ylim([0.5,sum(TrialsVector(:,1))+0.5])
set(gca,'fontsize',18);

% saveas(fig, fullfile(figurepath, strcat("AllAessionCombinedShock",'.jpg')));
%% Seperate UPS and PS
% for UPS
UPSIND = [1,3,4,7];
PSIND = [2,5,6,8];
excludedSession = [5];

CSTS_bin = ALLTimeStamp.CSTS_bin{1};

DF_ZScore_bin_UPS = [];
SP_ZScore_bin_UPS = [];
TrialsVector_UPS = [];
for i = 1:length(ALLTimeStamp.CSTS_bin)
    if ~any(i==excludedSession)
        try
            DF_ZScore_bin_UPS = vertcat(DF_ZScore_bin_UPS,ALLTimeStamp.DF_ZScore_bin{i}(UPSIND,:));
            SP_ZScore_bin_UPS = vertcat(SP_ZScore_bin_UPS,ALLTimeStamp.SP_ZScore_bin{i}(UPSIND,:));
            TrialsVector_UPS = vertcat(TrialsVector_UPS,ALLTimeStamp.Meta{i}.trialNum./2);
        catch
        end
    end
end
CSTrace1_UPS=(mean(DF_ZScore_bin_UPS,1)); % PSTH matrix (Z-score) mean
SPTrace1_UPS=(mean(SP_ZScore_bin_UPS,1)); % PSTH matrix (Z-score) mean for speed
CSTrace1SEM_UPS = (std(DF_ZScore_bin_UPS,1)./sqrt(size(DF_ZScore_bin_UPS,1)));
SPTrace1SEM_UPS = (std(SP_ZScore_bin_UPS,1)./sqrt(size(SP_ZScore_bin_UPS,1)));

DF_ZScore_bin_PS = [];
SP_ZScore_bin_PS = [];
TrialsVector_PS = [];
for i = 1:length(ALLTimeStamp.CSTS_bin)
    if ~any(i==excludedSession)
        try
            DF_ZScore_bin_PS = vertcat(DF_ZScore_bin_PS,ALLTimeStamp.DF_ZScore_bin{i}(PSIND,:));
            SP_ZScore_bin_PS = vertcat(SP_ZScore_bin_PS,ALLTimeStamp.SP_ZScore_bin{i}(PSIND,:));
            TrialsVector_PS = vertcat(TrialsVector_PS,ALLTimeStamp.Meta{i}.trialNum./2);
        catch
        end
    end
end
CSTrace1_PS=(mean(DF_ZScore_bin_PS,1)); % PSTH matrix (Z-score) mean
SPTrace1_PS=(mean(SP_ZScore_bin_PS,1)); % PSTH matrix (Z-score) mean for speed
CSTrace1SEM_PS = (std(DF_ZScore_bin_PS,1)./sqrt(size(DF_ZScore_bin_PS,1)));
SPTrace1SEM_PS = (std(SP_ZScore_bin_PS,1)./sqrt(size(SP_ZScore_bin_PS,1)));

%% Seperate UPS and PS plotting

% Z-value (subtract baseline) UPS
fig = figure("position",[100, 0, 2000, 600]);
ax(1) = subplot(2,4,1); hold on;
shadedErrorBar(CSTS_bin,CSTrace1_UPS,CSTrace1SEM_UPS,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,CSTrace1_UPS,'-k','LineWidth',2)
title('UPS', 'Fontsize', 18);
xlim([CSTS_bin(1) CSTS_bin(end)])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(CSTrace1_UPS));
CSmin=min(min(CSTrace1_UPS));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')
set(gca,'fontsize',18);

ax(2) = subplot(2,4,2); hold on;
shadedErrorBar(CSTS_bin,SPTrace1_UPS,SPTrace1SEM_UPS,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,SPTrace1_UPS,'-k','LineWidth',2)
title('Z-score PSTH', 'Fontsize', 18);
xlim([CSTS_bin(1) CSTS_bin(end)])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(SPTrace1_UPS));
CSmin=min(min(SPTrace1_UPS));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')
set(gca,'fontsize',18);

ax(3) = subplot(2,4,3); hold on;
yline(cumsum(TrialsVector_UPS(:,1))+0.5,'LineWidth',1.5)
imagesc(CSTS_bin,[1:size(DF_ZScore_bin_UPS,1)],DF_ZScore_bin_UPS,[-2,2]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([CSTS_bin(1) CSTS_bin(end)])
ylim([0.5,sum(TrialsVector_UPS(:,1))+0.5])
set(gca,'fontsize',18);

ax(4) = subplot(2,4,4); hold on;
yline(cumsum(TrialsVector_UPS(:,1))+0.5,'LineWidth',1.5)
imagesc(CSTS_bin,[1:size(SP_ZScore_bin_UPS,1)],SP_ZScore_bin_UPS,[-1,6]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([CSTS_bin(1) CSTS_bin(end)])
ylim([0.5,sum(TrialsVector_UPS(:,1))+0.5])
set(gca,'fontsize',18);

% Z-value (subtract baseline) PS
ax(5) = subplot(2,4,5); hold on;
shadedErrorBar(CSTS_bin,CSTrace1_PS, CSTrace1SEM_PS,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,CSTrace1_PS,'-k','LineWidth',2)
title('PS', 'Fontsize', 18);
xlim([CSTS_bin(1) CSTS_bin(end)])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(CSTrace1_PS));
CSmin=min(min(CSTrace1_PS));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')
set(gca,'fontsize',18);

ax(6) = subplot(2,4,6); hold on;
shadedErrorBar(CSTS_bin,SPTrace1_PS,SPTrace1SEM_PS,{'-b','color',[0,0,0]},0.5)
plot(CSTS_bin,SPTrace1_PS,'-k','LineWidth',2)
title('Z-score PSTH', 'Fontsize', 18);
xlim([CSTS_bin(1) CSTS_bin(end)])
xlabel('Time (s)', 'FontSize',18)
ylabel('Z-score','fontsize', 18);
CSmax=max(max(SPTrace1_PS));
CSmin=min(min(SPTrace1_PS));
ylim([CSmin CSmax*1.25]);
plot([0,0],[CSmin,CSmax*1.25],':r')
set(gca,'fontsize',18);

ax(7) = subplot(2,4,7); hold on;
yline(cumsum(TrialsVector_PS(:,1))+0.5,'LineWidth',1.5)
imagesc(CSTS_bin,[1:size(DF_ZScore_bin_PS,1)],DF_ZScore_bin_PS,[-2,2]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([CSTS_bin(1) CSTS_bin(end)])
ylim([0.5,sum(TrialsVector_PS(:,1))+0.5])
set(gca,'fontsize',18);

ax(8) = subplot(2,4,8); hold on;
yline(cumsum(TrialsVector_PS(:,1))+0.5,'LineWidth',1.5)
imagesc(CSTS_bin,[1:size(SP_ZScore_bin_PS,1)],SP_ZScore_bin_PS,[-1,6]);
ylabel('Trial #','fontsize', 18)
title([EventName{EventID},' Z-score PSTH'],'fontsize', 18)
xlim([CSTS_bin(1) CSTS_bin(end)])
ylim([0.5,sum(TrialsVector_PS(:,1))+0.5])
set(gca,'fontsize',18);

linkaxes([ax(1), ax(5)], 'y')
linkaxes([ax(2), ax(6)], 'y')

saveas(fig, fullfile(figurepath, strcat("UPSvsPSShock",'.jpg')));
%% different between UPS neural mean and UPS neural mean,
% 
% UPS_PS_DIFF = CSTrace1_UPS - CSTrace1_PS;
% fig = figure; hold on;
% plot(CSTS_bin,UPS_PS_DIFF,'-k','LineWidth',2)
% title('UPS', 'Fontsize', 18);
% xlim([CSTS_bin(1) CSTS_bin(end)])
% xlabel('Time (s)', 'FontSize',18)
% ylabel('Z-score','fontsize', 18);
% CSmax=max(max(UPS_PS_DIFF));
% CSmin=min(min(UPS_PS_DIFF));
% ylim([CSmin CSmax*1.25]);
% plot([0,0],[CSmin,CSmax*1.25],':r')
% set(gca,'fontsize',18);

DF_ZScore_bin_UPSPSDIFF = DF_ZScore_bin_UPS - DF_ZScore_bin_PS;
SP_ZScore_bin_UPSPSDIFF = SP_ZScore_bin_UPS - SP_ZScore_bin_PS;
TrialsINDEX = [0;cumsum(TrialsVector_PS(:,1))];

fig = figure; hold on;
for i = 1:length(TrialsINDEX) - 1
    subplot(1,2,1); hold on;
    UPSPSDIFF = mean(DF_ZScore_bin_UPSPSDIFF(TrialsINDEX(i)+1:TrialsINDEX(i+1),:));
    plot(CSTS_bin,UPSPSDIFF,'-k','LineWidth',1, 'color', [0.8,0.8,0.8])
    title('UPS-PS Neural diff', 'Fontsize', 18);
    xlim([CSTS_bin(1) CSTS_bin(end)])
    xlabel('Time (s)', 'FontSize',18)
    ylabel('Z-score','fontsize', 18);
    CSmax=max(max(DF_ZScore_bin_UPSPSDIFF));
    CSmin=min(min(DF_ZScore_bin_UPSPSDIFF));
    ylim([CSmin CSmax*1.25]);
    plot([0,0],[CSmin,CSmax*1.25],':r')
    set(gca,'fontsize',18);

    subplot(1,2,2); hold on;
    UPSPSDIFF = mean(SP_ZScore_bin_UPSPSDIFF(TrialsINDEX(i)+1:TrialsINDEX(i+1),:));
    plot(CSTS_bin,UPSPSDIFF,'-k','LineWidth',1, 'color', [0.8,0.8,0.8])
    title('UPS-PS Speed diff', 'Fontsize', 18);
    xlim([CSTS_bin(1) CSTS_bin(end)])
    xlabel('Time (s)', 'FontSize',18)
    ylabel('Z-score','fontsize', 18);
    CSmax=max(max(SP_ZScore_bin_UPSPSDIFF));
    CSmin=min(min(SP_ZScore_bin_UPSPSDIFF));
    ylim([CSmin CSmax*1.25]);
    plot([0,0],[CSmin,CSmax*1.25],':r')
    set(gca,'fontsize',18);
end
subplot(1,2,1); hold on;
UPSPSDIFF = mean(DF_ZScore_bin_UPSPSDIFF);
plot(CSTS_bin,UPSPSDIFF,'-k','LineWidth',2)
subplot(1,2,2); hold on;
UPSPSDIFF = mean(SP_ZScore_bin_UPSPSDIFF);
plot(CSTS_bin,UPSPSDIFF,'-k','LineWidth',2)

saveas(fig, fullfile(figurepath, strcat("UPSvsPSShockInEachSession",'.jpg')));