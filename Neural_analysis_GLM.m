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
    SessionsID = [1:length(files)-1];
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
    % Get event TS
    EventID = 1;
    EventTS = EventsON{EventID};

    % Find nearest Neural index
    CSidx=[];
    for i=1:length(EventTS)
        [~, CSidx(i,1)]=min(abs(NeuralTime(:,:)-EventTS(i)));
    end

    EventTimeSeries{EventID} = zeros(size(NeuralTime,1),1);
    EventTimeSeries{EventID}(CSidx) = 1;

    %% B-spline family
    FS_cam = 20;
    Pre = 0*FS_cam;
    Post = 5*FS_cam;
    KernelNum = 10;
    kernelTime = (-Pre:Post)./FS_cam;
    BsplineAll = spcol([0, 0, 0, linspace(0,1,KernelNum), 1, 1, 1], 4, linspace(0,1,Pre+Post+1));
    BsplineAll = spcol([0, 0, 0, linspace(0,1,KernelNum), 1, 1], 4, linspace(0,1,Pre+Post+1));
    BsplineAll = spcol([0, 0, 0, linspace(0,1,KernelNum), 1], 4, linspace(0,1,Pre+Post+1));
    figure;
    plot(BsplineAll);

    ConvTimeStamps = [];
    for i = 1:size(BsplineAll,2)
        ConvTimeStamps(:,i) = conv(EventTimeSeries{EventID},BsplineAll(:,i),'full');
    end
    ConvTimeStamps = ConvTimeStamps(1:length(NeuralTime),:);
    
    figure; hold on;
    plot(EventTimeSeries{EventID},"k",'LineWidth',2)
    plot(ConvTimeStamps)

    %% GLM and plot ezch contributions
    Dmatrix = [Com_speed,Com_denoised(:,3),ConvTimeStamps];
    y = Delat_Chan_GCamP_highpass;
    mdl = fitglm(Dmatrix,y)
    ypred = predict(mdl,Dmatrix);
%     ypred = mdl.Coefficients.Estimate' * [ones(size(Dmatrix,1),1),Dmatrix]';
    ypredSpeed = mdl.Coefficients.Estimate(2)' * [Dmatrix(:,1)]';
    ypredZaxis = mdl.Coefficients.Estimate(3)' * [Dmatrix(:,2)]';
    ypredShock = mdl.Coefficients.Estimate(4:end)' * [Dmatrix(:,3:end)]';
    
    figure; hold on;
    errorbar(mdl.Coefficients.Estimate,mdl.Coefficients.SE,"LineStyle","none")
    bar(mdl.Coefficients.Estimate)
    SigId = find(mdl.Coefficients.pValue < 0.05);
    plot(SigId,max(mdl.Coefficients.Estimate)*1.5,"r*")
   
    AllEstimate(session,:) = mdl.Coefficients.Estimate;

    Peak=max(Delat_Chan_GCamP_highpass);
    figure; hold on;
    plot(NeuralTime,Delat_Chan_GCamP_highpass, "DisplayName","Y")
    plot(NeuralTime,ypred, "DisplayName","ypred")
    plot(NeuralTime,ypredSpeed + mdl.Coefficients.Estimate(1), "DisplayName","ypredSpeed")
    plot(NeuralTime,ypredZaxis + mdl.Coefficients.Estimate(1), "DisplayName","ypredZaxis")
    plot(NeuralTime,ypredShock + mdl.Coefficients.Estimate(1), "DisplayName","ypredShock")
    plot(NeuralTime,EventTimeSeries{EventID}*Peak,"k",'LineWidth',2, "DisplayName","Event")
    plot(NeuralTime,ConvTimeStamps*Peak,'HandleVisibility','off')
    legend()
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
                        y=[Peak+Peak*(1+(k-1)), Peak+Peak*(2+(k-1)), Peak+Peak*(2+(k-1)), Peak+Peak*(1+(k-1))];
                        p1= patch(x,y,PatchColor(k,:),'FaceAlpha',0.5,'EdgeColor','none','HandleVisibility','off');
                    end
                end
            end
        end
    end

    figure; hold on;
    scatter(ypred,Delat_Chan_GCamP_highpass,1,ones(length(ypred),1),'DisplayName','TimeIndex')
    A = ypred'; B = Delat_Chan_GCamP_highpass';
    [rcof,pval]=corrcoef(A,B);
    text(0.2,0.7,['R = ',num2str(round(rcof(1,2),3))],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold');
    text(0.2,0.7-0.08,['p < 10^','{',num2str(ceil(log10(pval(2,1)))),'}'],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold')
    C=regress(B',[repmat(1,numel(A),1),A']);
    plot(linspace(min(A),max(A),11),linspace(min(A),max(A),11).*C(2)+C(1),'--','linewidth',3);
    title("Correlation between GLM pred and Neural")
    xlabel('ypred')
    ylabel('Delat_Chan_GCamP_highpass')
    
    AvgKernel(session,:) = mdl.Coefficients.Estimate(end-KernelNum+1:end)'*BsplineAll';
    figure; hold on;
    plot(kernelTime,AvgKernel(session,:))
    xlabel('Time (s)', 'FontSize',18)
    ylabel('Kernel','fontsize', 18);

%     %% save all figure
%     FolderName = figurepath;   % Your destination folder
%     FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
%     sessionName = file(1:end-4);
%     for iFig = 1:length(FigList)
%         FigHandle = FigList(iFig);
%         saveas(FigHandle, fullfile(FolderName, strcat(num2str(session,'%02.f'),"_", file,"_","GLM_analysis", num2str(iFig,'%02.f'),'.jpg')));
%     end
% 
%     close all
%     clearvars -except figurepath files session ProjectName Stimuli SessionsID AvgKernel
end

fig = figure; hold on;
shadedErrorBar(kernelTime, mean(AvgKernel),std(AvgKernel)./sqrt(size(AvgKernel,1)),{'-b','color',[0,0,0]},0.5)
saveas(fig, fullfile(figurepath, strcat("ShockKernelAvg",'.jpg')));

fig = figure; hold on;
errorbar(mean(AllEstimate),std(AllEstimate)./sqrt(size(AllEstimate,1)),"LineStyle","none")
bar(mean(AllEstimate))
saveas(fig, fullfile(figurepath, strcat("AllEstimateAvg",'.jpg')));
