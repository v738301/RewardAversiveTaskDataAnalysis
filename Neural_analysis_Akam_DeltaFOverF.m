function ProcessedNeuralStructure = Neural_analysis_Akam_DeltaFOverF(NeuralStructure,useAirPLS,wantplot)

%% Load data
NeuralTime = NeuralStructure.Time;
Chan_GCamP = NeuralStructure.Chan_GCamP;
Chan_Iso = NeuralStructure.Chan_Iso;

%% Data preprocssing
% Fill nans
Chan_GCamP = fillmissing(Chan_GCamP,'nearest');
Chan_Iso = fillmissing(Chan_Iso,'nearest');

fprintf('Resampling... \n')
% Resampling
FS = 12000;
resamplingFrequency = 1200;
% Express frequency as a ratio p/q.
[p, q] = rat(resamplingFrequency / FS);
% Resample: interpolate every p/q/f, upsample by p, filter, downsample by q.
[Chan_GCamP, time2] = resample(Chan_GCamP, NeuralTime, resamplingFrequency, p, q);
Chan_Iso = resample(Chan_Iso, NeuralTime, resamplingFrequency, p, q);
NeuralTime = time2;
FS = resamplingFrequency;

% Cut Start and end
CutLength = 2;
StartCut = FS*CutLength;
EndCut = FS*CutLength;
NeuralTime([1:StartCut,(end-EndCut+1):end]) = [];
Chan_GCamP([1:StartCut,(end-EndCut+1):end]) = [];
Chan_Iso([1:StartCut,(end-EndCut+1):end]) = [];

%% Both signal and control
if wantplot
    figure; hold on;
    plot(NeuralTime,Chan_GCamP);
    plot(NeuralTime,Chan_Iso);
    xlim([NeuralTime(1), NeuralTime(end)]);
    xlabel('Time (seconds)');
    title('Signal vs Control');
    legend({'GCamP','Isobestic'})
end
%% Denoising and lowpassed
fprintf('Denoising... \n')
% median filtering
Chan_GCamP_denoised=medfilt1(Chan_GCamP,10,'truncate');
Chan_Iso_denoised=medfilt1(Chan_Iso,10,'truncate');

% Filter out high frequency noise which higher than 10 hz ==> (-inf ~ 10 Hz)
CutOff = 10; % Hz
[b,a] = butter(2, CutOff/(FS/2),'low');
Chan_GCamP_denoised = filtfilt(b,a, Chan_GCamP_denoised);
Chan_Iso_denoised = filtfilt(b,a, Chan_Iso_denoised);

% Chan_GCamP_denoised (-inf ~ 10 Hz)
% Chan_Iso_denoised (-inf ~ 10 Hz)

%% Fit before photobleachinf correction
bls=polyfit(Chan_Iso_denoised(1:end),Chan_GCamP_denoised(1:end),1);
Estimated_motion_Bble=bls(1).*Chan_Iso_denoised+bls(2);
MotionCorrected_GCamP_Bble = Chan_GCamP_denoised(:)-Estimated_motion_Bble(:);
if wantplot
    figure; hold on;
    plot(NeuralTime,Chan_GCamP_denoised);
    plot(NeuralTime, Estimated_motion_Bble);
    plot(NeuralTime, MotionCorrected_GCamP_Bble);
    xlim([NeuralTime(1), NeuralTime(end)]);
    xlabel('Time (seconds)');
    title('Signal vs Fitted Control BEFORE bleaching correction');
    legend({'GCamP','Fitted Signal','Estimated_motion_Bble'})
end
%% Photobleaching correction
fprintf('Photobleaching correction... \n')
% filter out lower signal related to photobleaching ==> (0.1 ~ 10 Hz)
useAirPLS = 0;
expFit = 1;
% airPLSPara = {5e9, 2, 0.1, 0.5, 50};
airPLSPara = {1e15, 2, 0.1, 0.5, 50};
if useAirPLS
    [~, signalBaseline] = airPLS(Chan_GCamP_denoised', airPLSPara{:});
    signalBaseline = signalBaseline';
    Chan_GCamP_highpass = Chan_GCamP_denoised - signalBaseline;
    [~, referenceBaseline] = airPLS(Chan_Iso_denoised', airPLSPara{:});
    referenceBaseline = referenceBaseline';
    Chan_Iso_highpass = Chan_Iso_denoised - referenceBaseline;
elseif expFit == 1
    signalFit = fit(NeuralTime, Chan_GCamP_denoised,'exp2');
    signalBaseline1 = signalFit(NeuralTime);
    Chan_GCamP_highpass = Chan_GCamP_denoised - signalBaseline1;
    signalFit = fit(NeuralTime, Chan_Iso_denoised, 'exp2');
    signalBaseline2 = signalFit(NeuralTime);
    Chan_Iso_highpass = Chan_Iso_denoised - signalBaseline2;
else
    CutOff = 0.0005; % Hz
    [b,a] = butter(2, CutOff/(FS/2),'high');
    Chan_GCamP_highpass = filtfilt(b,a, Chan_GCamP_denoised);
    Chan_Iso_highpass = filtfilt(b,a, Chan_Iso_denoised);
end
% Chan_GCamP_highpass (0.1 ~ 10 Hz)
% Chan_Iso_highpass (0.1 ~ 10 Hz)

%% Baseline
CutOff = 0.001; % Hz
[b,a] = butter(2, CutOff/(FS/2),'low');
Chan_GCamP_Baseline = filtfilt(b,a, Chan_GCamP_denoised);

% Chan_GCamP_Baseline (-inf ~ 0.001)

%% Plot High passed, Denoised signal and Raw signal
if wantplot
    figure; hold on;
    plot(NeuralTime,Chan_GCamP);
    plot(NeuralTime,Chan_Iso);
    plot(NeuralTime,Chan_GCamP_denoised);
    plot(NeuralTime,Chan_Iso_denoised);
    plot(NeuralTime,signalBaseline1);
    plot(NeuralTime,signalBaseline2);
    plot(NeuralTime,Chan_GCamP_highpass);
    plot(NeuralTime,Chan_Iso_highpass);
    xlim([NeuralTime(1), NeuralTime(end)]);
    xlabel('Time (seconds)');
    title('Signal vs Control');
    legend({'GCamP','Isobestic','GCamP-Denoised','Isobestic-Denoised','FitGCam','FitIso','GCamP-Highpassed','Isobestic-Highpassed'})
end
%% Fitted Signal
A = Chan_Iso_highpass(:)';
B = Chan_GCamP_highpass(:)';

if wantplot
    figure; hold on;
    scatter(A(1:1200:end), B(1:1200:end));
    [rcof,pval]=corrcoef(A,B);
    text(0.2,0.9,['R = ',num2str(round(rcof(1,2),3))],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold');
    text(0.2,0.9-0.08,['p < 10^','{',num2str(ceil(log10(pval(2,1)))),'}'],'units','normalized','FontSize',12,'HorizontalAlignment','left','FontWeight','bold')
    C=regress(B',[repmat(1,numel(A),1),A']);
    plot(linspace(min(A),max(A),11),linspace(min(A),max(A),11).*C(2)+C(1),'--','linewidth',3);
end

bls=polyfit(Chan_Iso_highpass(1:end),Chan_GCamP_highpass(1:end),1);
Estimated_motion=bls(1).*Chan_Iso_highpass+bls(2);
MotionCorrected_GCamP = Chan_GCamP_highpass(:)-Estimated_motion(:);

% MotionCorrected_GCamP (0.1 ~ 10 Hz) minus motion power

if wantplot
    figure; hold on;
    plot(NeuralTime,Chan_GCamP_highpass);
    plot(NeuralTime, Estimated_motion);
    plot(NeuralTime, MotionCorrected_GCamP);
    xlim([NeuralTime(1), NeuralTime(end)]);
    xlabel('Time (seconds)');
    title('Signal vs Fitted Control');
    legend({'GCamP','Fitted Signal','Corrected Signal'})
end
%% Delta F/F
% MotionCorrected_GCamP (0.1 ~ 10 Hz) minus motion power
% Chan_GCamP_Baseline (-inf ~ 0.001)

Delat_Chan_GCamP_highpass = MotionCorrected_GCamP./Chan_GCamP_Baseline;
%% ALL signal note
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

ProcessedNeuralStructure = struct();
ProcessedNeuralStructure.FS = FS;
ProcessedNeuralStructure.NeuralTime = NeuralTime;
ProcessedNeuralStructure.Chan_GCamP_denoised = Chan_GCamP_denoised;
ProcessedNeuralStructure.Chan_Iso_denoised = Chan_Iso_denoised;
ProcessedNeuralStructure.Estimated_motion_Bble = Estimated_motion_Bble;
ProcessedNeuralStructure.MotionCorrected_GCamP_Bble = MotionCorrected_GCamP_Bble;
ProcessedNeuralStructure.Chan_GCamP_highpass = Chan_GCamP_highpass;
ProcessedNeuralStructure.Chan_Iso_highpass = Chan_Iso_highpass;
ProcessedNeuralStructure.Estimated_motion = Estimated_motion;
ProcessedNeuralStructure.MotionCorrected_GCamP = MotionCorrected_GCamP;
ProcessedNeuralStructure.Chan_GCamP_Baseline = Chan_GCamP_Baseline;

fprintf('Done! \n')
end

