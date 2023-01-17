function AllStructure = Read3structure(file,AllPath)

%% SET PATHS
datapathDORIC = AllPath(1);
datapathBEH = AllPath(2);
datapathTRACK = AllPath(3);

%% Output struct
AllStructure = struct();
AllStructure.meta = {};

%% Get Doric data
fullname = fullfile(datapathDORIC,file);
AllStructure.meta{1} = file;
fprintf('Read %s data to memory \n',file)
NeuralStructure = ReadDoric(fullname);

%% Get BEH data
AnimalName = file(1:4);
Date = char(datetime(file(6:13),'InputFormat','yyyyMMdd','Format','yyyy-MM-dd'));
SelectTarget = string([Date,'*', AnimalName,'*']);

filename = 'Events.nev';
[datapath_temp] = dir(fullfile(datapathBEH,SelectTarget));
if numel(datapath_temp) > 1
    datapath_temp.name
    prompt = "Which BEH file you want? \n";
    x = input(prompt);
    datapath = fullfile(datapath_temp(x).folder,datapath_temp(x).name);
    names = datapath_temp(x).name;
elseif numel(datapath_temp) == 0
    error('Error. \n No Behavior files found \n')
else
    datapath = fullfile(datapath_temp.folder,datapath_temp.name);
    names = datapath_temp.name;
end
fullname = [datapath,'\',filename];
AllStructure.meta{2} = names;
fprintf('Read %s data to memory \n',names)
BehStrcture = read_nev(fullname);

%% Get tracking data
Date = char(datetime(file(6:13),'InputFormat','yyyyMMdd','Format','yyyy_MM_dd'));
SelectTarget = string([AnimalName,'_',Date,'*']);

filename = 'COM\predict_results\com3d.mat';
[datapath_temp] = dir(fullfile(datapathTRACK,SelectTarget));
if numel(datapath_temp) > 1
    datapath_temp.name
    prompt = "Which TRACK file you want? \n";
    x = input(prompt);
    datapath = fullfile(datapath_temp(x).folder,datapath_temp(x).name);
    names = datapath_temp(x).name;
elseif numel(datapath_temp) == 0
    error('Error. \n No Tracking files found \n')
else
    datapath = fullfile(datapath_temp.folder,datapath_temp.name);
    names = datapath_temp.name;
end
fullname = [datapath,'\',filename];
AllStructure.meta{3} = names;
fprintf('Read %s data to memory \n',names)
TrackStrcture = load(fullname);

%% Output
AllStructure.NeuralStructure = NeuralStructure;
AllStructure.BehStrcture = BehStrcture;
AllStructure.TrackStrcture = TrackStrcture;
