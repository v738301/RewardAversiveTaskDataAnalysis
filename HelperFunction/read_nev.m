% clear all
% close all
% datapath = '/Volumes/KUNLINDISK/2022-12-26_17-02-48_B634/Events.nev';
function DataStrcture = read_nev(datapath)

%%
FieldSelection(1:5) = 1;
ExtractHeader = 0;
ExtractMode = 1;
[TimeStamp, ~, ~, ~, EventStrings] = Nlx2MatEV(datapath, FieldSelection, ExtractHeader, ExtractMode, []);

%%
[TTLtype,~,ic] = unique(EventStrings);
startTime = TimeStamp(1);
Timestamps = TimeStamp - startTime;
Timestamps = Timestamps./(10^6); %% Micro to Seconds

%% build bincode library
binCodelib = [];
for i = 1:length(TTLtype)
    CodeString = TTLtype{i};
    if ~isempty(findstr(CodeString,'port 1 value (0x00'))
        code_temp0 = hexmat2bin(CodeString(53));
        code_temp1 = hexmat2bin(CodeString(54));
        for k = 1:length(code_temp0{1})
            binCodelib(i,k) = str2num(code_temp0{1}(k));
            binCodelib(i,k+4) = str2num(code_temp1{1}(k));
        end
    elseif ~isempty(findstr(CodeString,'port 2 value (0x00'))
        code_temp0 = hexmat2bin(CodeString(53));
        code_temp1 = hexmat2bin(CodeString(54));
        for k = 1:length(code_temp0{1})
            binCodelib(i,k) = str2num(code_temp0{1}(k));
            binCodelib(i,k+4) = str2num(code_temp1{1}(k));
        end
    else
    end
end
%%
clear binCode1 binCode2 Timestamp1 Timestamp2
binCode1 = nan(size(ic,1),8); ind1 = 1;
binCode2 = nan(size(ic,1),8); ind2 = 1;
Timestamp1 = nan(size(ic,1),1);
Timestamp2 = nan(size(ic,1),1);
for i = 1:length(ic)
    %     if mod(i,10000) == 0
    %         sprintf([int2str(i),'/',int2str(length(ic))])
    %     end
    CodeString = TTLtype(ic(i));
    if ~isempty(findstr(CodeString{1},'port 1 value (0x00'))
        binCode1(ind1,:) = binCodelib(ic(i),:);
        Timestamp1(ind1,:) = Timestamps(i);
        ind1 = ind1 + 1;
    elseif ~isempty(findstr(CodeString{1},'port 2 value (0x00'))
        binCode2(ind2,:) = binCodelib(ic(i),:);
        Timestamp2(ind2,:) = Timestamps(i);
        ind2 = ind2 + 1;
    else
    end
end
binCode1(sum(isnan(binCode1),2)>0,:) = [];
binCode2(sum(isnan(binCode2),2)>0,:) = [];
Timestamp1(sum(isnan(Timestamp1),2)>0) = [];
Timestamp2(sum(isnan(Timestamp2),2)>0) = [];

% hexmat2bin('0') --> 0000
% hexmat2bin('1') --> 0001
% hexmat2bin('2') --> 0010
% hexmat2bin('3') --> 0011
% hexmat2bin('4') --> 0100
% hexmat2bin('5') --> 0101
% hexmat2bin('6') --> 0110
% hexmat2bin('7') --> 0111
% hexmat2bin('8') --> 1000
% hexmat2bin('9') --> 1001
% hexmat2bin('A') --> 1010
% hexmat2bin('B') --> 1011
% hexmat2bin('C') --> 1100
% hexmat2bin('D') --> 1101
% hexmat2bin('E') --> 1110
% hexmat2bin('F') --> 1111

% Code1
% None Cam3 Cam2 Cam1 Trig IR1 IR2 None
% Code2
% Shock Sound3 Sound2 Sound1 Gen2 None WP1 WP2

%% TTl Timestamps
% aviod TTL open in first timestamp
binCode1 = [zeros(1,size(binCode1,2));binCode1];
binCode2 = [1,zeros(1,size(binCode2,2)-1);binCode2];

% TTL ON
indforcam1ON = find(diff(binCode1(:,2))==1);
indforcam2ON = find(diff(binCode1(:,3))==1);
indforcam3ON = find(diff(binCode1(:,4))==1);
camtriggerON = find(diff(binCode1(:,5))==1);
IR1INDON = find(diff(binCode1(:,6))==1);
IR2INDON = find(diff(binCode1(:,7))==1);

ShockINDON = find(diff(binCode2(:,1))==-1); %% shock was coded inversely
Sound1INDON = find(diff(binCode2(:,4))==1);
WP1INDON = find(diff(binCode2(:,7))==1);
WP2INDON = find(diff(binCode2(:,8))==1);

Cam1ON = Timestamp1(indforcam1ON);
Cam2ON = Timestamp1(indforcam2ON);
Cam3ON = Timestamp1(indforcam3ON);
CamTrigON = Timestamp1(camtriggerON);
IR1ON = Timestamp1(IR1INDON);
IR2ON = Timestamp1(IR2INDON);

ShockON = Timestamp2(ShockINDON);
Sound1ON = Timestamp2(Sound1INDON);
WP1ON = Timestamp2(WP1INDON);
WP2ON = Timestamp2(WP2INDON);

% TTL OFF
indforcam1OFF = find(diff(binCode1(:,2))==-1);
indforcam2OFF = find(diff(binCode1(:,3))==-1);
indforcam3OFF = find(diff(binCode1(:,4))==-1);
camtriggerOFF = find(diff(binCode1(:,5))==-1);
IR1INDOFF = find(diff(binCode1(:,6))==-1);
IR2INDOFF = find(diff(binCode1(:,7))==-1);

ShockINDOFF = find(diff(binCode2(:,1))==1); %% shock was coded inversely
Sound1INDOFF = find(diff(binCode2(:,4))==-1);
WP1INDOFF = find(diff(binCode2(:,7))==-1);
WP2INDOFF = find(diff(binCode2(:,8))==-1);

Cam1OFF = Timestamp1(indforcam1OFF);
Cam2OFF = Timestamp1(indforcam2OFF);
Cam3OFF = Timestamp1(indforcam3OFF);
CamTrigOFF = Timestamp1(camtriggerOFF);
IR1OFF = Timestamp1(IR1INDOFF);
IR2OFF = Timestamp1(IR2INDOFF);

ShockOFF = Timestamp2(ShockINDOFF);
Sound1OFF = Timestamp2(Sound1INDOFF);
WP1OFF = Timestamp2(WP1INDOFF);
WP2OFF = Timestamp2(WP2INDOFF);

%% put variable into data structure
DataStrcture = struct();
DataStrcture.Cam1ON = Cam1ON;
DataStrcture.Cam2ON = Cam2ON;
DataStrcture.Cam3ON = Cam3ON;
DataStrcture.CamTrigON = CamTrigON;
DataStrcture.IR1ON = IR1ON;
DataStrcture.IR2ON = IR2ON;
DataStrcture.ShockON = ShockON;
DataStrcture.Sound1ON = Sound1ON;
DataStrcture.WP1ON = WP1ON;
DataStrcture.WP2ON = WP2ON;

DataStrcture.Cam1OFF = Cam1OFF;
DataStrcture.Cam2OFF = Cam2OFF;
DataStrcture.Cam3OFF = Cam3OFF;
DataStrcture.CamTrigOFF = CamTrigOFF;
DataStrcture.IR1OFF = IR1OFF;
DataStrcture.IR2OFF = IR2OFF;
DataStrcture.ShockOFF = ShockOFF;
DataStrcture.Sound1OFF = Sound1OFF;
DataStrcture.WP1OFF = WP1OFF;
DataStrcture.WP2OFF = WP2OFF;

% make On and Off timestamps same length by delete On timestamp (avoid ON longer than OFF)
names = fieldnames(DataStrcture);
for i = 1:length(names)
    if ~isempty(strfind(names{i},'ON'))
        % check length
        lenDiff = length(DataStrcture.(names{i})) - length(DataStrcture.([names{i}(1:end-2),'OFF']));
        if lenDiff > 0
            if DataStrcture.(names{i})(end) - DataStrcture.([names{i}(1:end-2),'OFF'])(end) > 0
                DataStrcture.(names{i})(end-lenDiff+1:end) = [];
            else
                msg = 'Last ON is latter than last OFF';
                error(msg)
            end
        end
        % check ON always earlier than OFF
        timeDelays = [];
        timeDelays = DataStrcture.(names{i}) - DataStrcture.([names{i}(1:end-2),'OFF']);
        if any(timeDelays>0)
            msg = 'Some ONs is latter than their OFFs';
            error(msg)
        end
    end
end

DataStrcture.TotalLen = numel(TimeStamp);

end
