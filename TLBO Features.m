%% TLBO + LPQ Image Feature Selection
% This code extract LPQ features out of 5 classes of images and selects best 
% desire number of features using TLBO algorithm. 

%% Making Things Ready !!!
clc;
clear; 
warning('off');

%% LPQ Feature Extraction
% Read input images
path='Dat';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);
end;
% Color to Gray Conversion
for i = 1 : filesnumber(1,1)
images{i}=rgb2gray(images{i});
disp(['To Gray :   ' num2str(i) ]);end;
% Contrast Adjustment
for i = 1 : filesnumber(1,1)
adjusted2{i}=imadjust(images{i});
disp(['Contrast Adjust :   ' num2str(i) ]);end;
% Resize Image
for i = 1 : filesnumber(1,1)
resized2{i}=imresize(adjusted2{i}, [256 256]);
disp(['Image Resized :   ' num2str(i) ]);end;

%% LPQ Features
clear LPQ_tmp;clear LPQ_Features;

winsize=9;

for i = 1 : filesnumber(1,1)
LPQ_tmp{i}=lpq(resized2{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)
LPQ_Features(i,:)=LPQ_tmp{i};end;

%% Labeling for Classification
sizefinal=size(LPQ_Features);
sizefinal=sizefinal(1,2);
%
LPQ_Features(1:10,sizefinal+1)=1;
LPQ_Features(11:20,sizefinal+1)=2;
LPQ_Features(21:30,sizefinal+1)=3;
LPQ_Features(31:40,sizefinal+1)=4;
LPQ_Features(41:50,sizefinal+1)=5;

% ------------------------------------------------------
%% Feature Selection
% Data Preparation
x=LPQ_Features(:,1:end-1)';
t=LPQ_Features(:,end)';
data.x=x;
data.t=t;
data.nx=size(x,1);
data.nt=size(t,1);
data.nSample=size(x,2);

%% Number of Desired Features
nf=32;

%% Cost Function
CostFunction=@(u) FeatureCost(u,nf,data);
% Number of Decision Variables
nVar=data.nx;
% Size of Decision Variables Matrix
VarSize=[1 nVar];
% Lower Bound of Variables
VarMin=0;
% Upper Bound of Variables
VarMax=1;

%--------------------------------------------------------------
%% TLBO Parameters
MaxIt = 30;        
nPop = 2;          
empty_individual.Position = [];
empty_individual.Cost = [];
empty_individual.Out = [];
pop = repmat(empty_individual, nPop, 1);
BestSol.Cost = inf;
for i = 1:nPop
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
[pop(i).Cost pop(i).Out] = CostFunction(pop(i).Position);
if pop(i).Cost < BestSol.Cost
BestSol = pop(i);
end
end
BestCost = zeros(MaxIt, 1);
%% TLBO Body
for it = 1:MaxIt
Mean = 0;
for i = 1:nPop
Mean = Mean + pop(i).Position;
end
Mean = Mean/nPop;
% Select Teacher
Teacher = pop(1);
for i = 2:nPop
if pop(i).Cost < Teacher.Cost
Teacher = pop(i);
end
end
% Teacher 
for i = 1:nPop
newsol = empty_individual;
% Teaching Factor
TF = randi([1 2]);
% Teaching (moving towards teacher)
newsol.Position = pop(i).Position ...
+ rand(VarSize).*(Teacher.Position - TF*Mean);
% Clipping
newsol.Position = max(newsol.Position, VarMin);
newsol.Position = min(newsol.Position, VarMax);
% Evaluation
[newsol.Cost newsol.Out] = CostFunction(newsol.Position);
% Comparision
if newsol.Cost<pop(i).Cost
pop(i) = newsol;
if pop(i).Cost < BestSol.Cost
BestSol = pop(i);
end
end
end
% Learner 
for i = 1:nPop
A = 1:nPop;
A(i) = [];
j = A(randi(nPop-1));
Step = pop(i).Position - pop(j).Position;
if pop(j).Cost < pop(i).Cost
Step = -Step;
end
newsol = empty_individual;
% Teaching (moving towards teacher)
newsol.Position = pop(i).Position + rand(VarSize).*Step;
% Clipping
newsol.Position = max(newsol.Position, VarMin);
newsol.Position = min(newsol.Position, VarMax);
% Evaluation
[newsol.Cost newsol.Out]= CostFunction(newsol.Position);
% Comparision
if newsol.Cost<pop(i).Cost
pop(i) = newsol;
if pop(i).Cost < BestSol.Cost
BestSol = pop(i);
end
end
end
% Store Record for Current Iteration
BestCost(it) = BestSol.Cost;
disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
end

% Plot ---------------------------------------------- 
plot(BestCost, '--k','linewidth',2);
xlabel('Iteration');
ylabel('Bees Cost');

%---------------------------------------------------
%% Creating Features Matrix
% Extracting Data
RealData=data.x';
% Extracting Labels
RealLbl=data.t';
FinalFeaturesInd=BestSol.Out.S;
% Sort Features
FFI=sort(FinalFeaturesInd);
% Select Final Features
Bio_Features=RealData(:,FFI);
% Adding Labels
Bio_Features_Lbl=Bio_Features;
Bio_Features_Lbl(:,end+1)=RealLbl;
LPQ_Bio=Bio_Features_Lbl;
% Plot
figure;
plot(FinalFeaturesInd, 'linewidth' , 2);
title('Selected Features');
xlabel ('Feature Number');
ylabel ('Feature Index');



