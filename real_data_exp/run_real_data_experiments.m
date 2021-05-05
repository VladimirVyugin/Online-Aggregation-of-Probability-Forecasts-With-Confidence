% 
% This file includes code for the paper 
% Online Aggregation of Probability Forecasts with Confidence. 
% @authors: Vladimir Vyugin, Vladimir Trunov
% 

function [R] = run_real_data_experiments()
Mode = 'LongList';    
for variant = 2:4
    R = explore_elf_data(variant,Mode);
end

tag = '21/long/';   
[Totals] = draw_elf_results(tag);

% figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R] = explore_elf_data(variant,eMode)
% To create FIG and EPS figures, set paper_figures to true!
if nargin==0
    variant = 2;    
    %eMode = 'ShortList';
    eMode = 'LongList';
    %%       TRAIN       TEST(Pit)    MIX
    % Var 1; soft_mode  UNIFORM  sOSOFT_NODEGSet.uniform_conf_level = 1;
    % Var 2; SOFT    the_soft_mode = 1, GSet.uniform_conf_level = 0;
    % Var 3; HARD    GSet.uniform_conf_level = 0;
  
    % variant == 4; the_soft_mode = 0, GSet.uniform_conf_level = 1;
end
figure_path = '..\figures2\';

variantNames = {'SoftTrain+UniformTest', 'HardTrain+SoftTest',...
    'HardTrain+HardTest', 'HardTrain+UniformTest','WideTrain+SoftTest',...
    'SoftTrain+SoftTest'};
    
%% 1. %%%%%%%%%%%%%%% I N I T I A L   S E T T I N G S   %%%%%%%%%%%%%%%(36)
LineSpecList = {'-b','-g','-k','-r','--b','--g','--r','--k',':b',':g',':r'};
verbose = 1;        % Show details
alpha = 0.001;      % FS parameter

% GSet - default 
    GSet.debug   = 0;
    GSet.verbose = 0; % Show and save scatterplots 
    GSet.variant = variant; 
    GSet.a = 0;                                  % Min load value (MHW)   
    GSet.b = 500;                                % Max load value (MHW)    
    GSet.T_min = 0;                              % Min temperature level (F)
    GSet.T_max = 110;                            % Max temperature level (F)
    GSet.L_grid = (GSet.a:1:GSet.b);             % Load grid
    GSet.T_grid = (GSet.T_min:1:GSet.T_max);     % Temperature grid
    a = GSet.a;                                  % Min load value (MHW)   
    b = GSet.b;   
    
% ConfSet default 
    ConfSet.z_power = 1;  % Nonlinear comp_z transformation
    ConfSetInit.action = 'None'; %('Train' 'Test')
    ConfSetInit.hard_train_min_level = 0.99;  
    ConfSetInit.soft_style = 'ByLevel'; %'Random'
    ConfSetInit.soft_train_min_level = 0; 
    
    ConfSetInit.sat_level  = 0.5; 
    ConfSetInit.neg_level  = 0.25; 
    ConfSetInit.z_power = 1;
%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% variants setting trainsample
if variant==1 %% {'SoftTrain+UniformTest'};
    ConfSet = ConfSetInit;
    ConfSet.train_mode = 'SOFT';    % SOFT (1-HARD 2-SOFT 3-WIDE
    ConfSet.aggr_mode  = 'Uniform'; % Uniform (1-Hard,2-Soft,3-Uniform)
    if strcmp(ConfSet.aggr_mode,'Uniform')
        % Все P_t = 1. Forced to be 1
        ConfSet.uniform_conf_level = 1;   % Все P_t = 1. Forced to be 1
    else
        error('In setting params...')
    end
    w_dir = 'variant1';
    q=1;
    variantNames{variant} = 'SoftTrain+UniformTest'

elseif variant==2  %'HardTrain+SoftTest' 
    ConfSet = ConfSetInit;
    ConfSet.train_mode = 'HARD';    % SOFT (1-HARD 2-SOFT 3-WIDE
    ConfSet.aggr_mode  = 'Soft';    % Soft (1-Hard,2-Soft,3-Uniform)
    ConfSet.uniform_conf_level = 0;
    ConfSet.neg_level  = 0.25;   %0.30; 
    %ConfSet.neg_level  =  0.1;   
    w_dir = 'variant2';
    variantNames{variant} = 'HardTrain+SoftTest';
elseif variant==3  %'HardTrain+HardTest' 
    ConfSet = ConfSetInit;
    ConfSet.train_mode = 'HARD';     % SOFT (1-HARD 2-SOFT 3-WIDE
    ConfSet.aggr_mode  = 'Hard';     % Soft (1-Hard,2-Soft,3-Uniform)
    ConfSet.uniform_conf_level = 0;  
    w_dir = 'variant3'; % "Sleeping" experts
    variantNames{variant} = 'HardTrain+HardTest';

elseif variant==4 % 'HardTrain+UniformTest'};
    ConfSet = ConfSetInit;
    ConfSet.train_mode = 'HARD';    % SOFT (1-HARD 2-SOFT 3-WIDE
    ConfSet.aggr_mode  = 'Uniform'; % Uniform (1-Hard,2-Soft,3-Uniform)
    ConfSet.uniform_conf_level = 1; % All P_t = 1. Forced to be 1
    w_dir = 'variant4'; 
    variantNames{variant} = 'HardTrain+UniformTest';
else
    error('Illegal variant!');
end

if GSet.verbose
    [r,str]=txt_write_line('elf_train_log.txt',['V-' ...
    num2str(GSet.variant) ': ' datestr(now) '  Mix:' ConfSet.aggr_mode ' Soft:' ConfSet.soft_style],[])
end

ConfSet.min_conf_level = 0;  %% 0.0000001;   %debug       
GSet.UseAveTemp = 0;   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2. Initial settings
FT_Predictor = 'AT_1';  % Farenhate Temperature Predictor
frc_list = create_expert_list(eMode);

data_folder_path = 'elf_data\';         
q=1;
M = length(frc_list);   %% Num experts 

%% 3. %%%%% 
for ii=[3]
    load([data_folder_path 'Train14.mat']); 
    q=1;
end

%4. Create array of experts using frc_list  
for ii=[4]
    comp_v=[];   % Competence index column (LOGICAL) for current expert
    comp_z=[];   % Competence score vector-DOUBLE/[0 1] for current expert
    conf_matrix = [];
    tikers={};   
    for alg=1:M
        frc    = frc_list{alg}
        tikers = [tikers [num2str(alg) '. ' frc.descr]];
    end
    M = length(frc_list);   
    cmS_list = cell(1,M);  
end

% 5. Responce variable as a literal
varY = cell(1,1);  varY(1)  = {'Load'};     
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 6. Train M experts
ind_list = cell(M,1);
for alg = 1:M
    frc = frc_list{alg}  
    
    % Training 
    [cmS,ind,confid] = train_ds_forecaster(frc,ds_train,ds_event_train,varY,GSet,ConfSet); 
    conf_matrix = [conf_matrix confid];
    ind_list{alg} = ind;
    
    trainT = table2array(ds_train(:,'tSerial')); 
    Y = table2array(ds_train(:,varY));   
    ConfSet.action ='train';
    [Yh,comp_z] = apply_ds_forecaster(cmS,ds_train,ds_event_train,[],GSet,ConfSet);
    cmS_list{alg} = cmS;   
end

% % Save results in "TrainedModels.mat"
% if GSet.debug
%     save([w_dir '/TrainedModels.mat'],'cmS_list','ind_list','GSet','ConfSet');
%     save([w_dir '/train_conf_matrix.mat'],'conf_matrix');
% end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 8. LOAD_FULL_DATA_SAMPLE
for LOAD_FULL_DATA = 1   % Загружаем все данные
    % Загружаем ds_meteo ds_event_meteo ds_test ds_event_test
    load([data_folder_path 'Test14.mat']);

    t1 = 0;          % Meteo part (deleted)
    t2 = 1;          % Begin of train part
    t3 = 52583+1;    % Begin of test part
    t4 = 52583+8017; % 60600 end of test part
    cPoints = [t1 t2 t3 t4];
    ds_full = [ds_train; ds_test];
    ds_event_full = [ds_event_train; ds_event_test];
    
    show_confidence_levels = false;  
    if show_confidence_levels 
        ConfSet.action ='test';
        [r] = show_prior_confidence(frc_list,ds_event_test,GSet,ConfSet);
    end
end

%% 9. Testing and aggregation
for ii=[9]
    conf_matrix   = [];
    WWwarm_CRPS   = [];  % Warm weights
    WWvovk_CRPS   = [];  % Wovk weights
    W_it_Warm_CRPS = ones(1,M)/M;    
    W_it_Vovk_CRPS = ones(1,M)/M;    
    WWetta_LSQ2    = [];  
    W_it_etta_LSQ2 = ones(1,M)/M;   
    LL_meas        = [];            
    LL_pred        = [];            

    iStart  = cPoints(3); 
    iFinish = cPoints(4);    
    
    rST = 0;                   
    ST_CRPS = [];              
    SVL_W_CRPS = [];             % Stored Virtual Losses by Warmuth  L x M
    SVL_V_CRPS = [];             % Stored Virtual Losses by Vovk     L x M
    inst_vl_W_CRPS = zeros(1,M); % InstantVirtualLosses by Warmuth   1 x M
    inst_vl_V_CRPS = zeros(1,M); % InstantVirtualLosses by Vovk      1 x M  
    ST_LSQ2 = [];              
    SVL_etta_LSQ2 = [];             % Stored Virtual LSQ2 Losses
    inst_vl_etta_LSQ2 = zeros(1,M); % InstantVirtual LSQ2Losses
    P_it_Stored = [];          
    hour_step = 1;             
    iShift = 0;      
    aa3D_cdf=[];                % For cumulating 3D pictures
    aa3D_pdf=[];
    ConfSet.action ='test';
        
    if variant==2   % just for fig6l!
        draw_diagram = false;   % just for fig6l!
        if draw_diagram
            Pt = [];
            ds_event_test = ds_event_full(iStart:iFinish,:);
            T = iFinish-iStart+1;
            test_conf_matrix = [];
            for m = 1:M
                frc = frc_list{m};
                [var_list,confid]=get_sample_conf(frc,ds_event_test,GSet,ConfSet); 
                test_conf_matrix = [test_conf_matrix confid];
            end
            bw =1.2;
            Pt = nan(T,length(frc_list));
            for i = length(frc_list):-1:2
                frc = frc_list{i};
                Pt(:,i) = test_conf_matrix(:,i)+(i*bw);
            end 
           
            h=figure;
            plot(Pt(:,2:M));
            xlim([0 T+1]);
            ylim([2.2 27]);
            saveas(h,'w_dir\test_conf_matrix.png');
            save([w_dir '\test_conf_matrix.mat'],'test_conf_matrix');
        end
    end
    
    for i = iStart+iShift : hour_step: iFinish  
    crps_row = nan(1,M+2);      % Current losses
    rST = rST+1;                % Line number
    
    Yactual  = table2array(ds_full(i,cmS.varY));  % Actual Load Value
    DTserial = ds_full.tSerial(i);                % Time Instant
    % CDF are stored in Fh_list just for current time instant
    Fh_list = cell(1,length(cmS_list));           
    PointF = nan(1,M); 
    P_it = zeros(1,M); 
    for alg = 1:M      
        frc = frc_list{alg};    % Current expert description
        cmS = cmS_list{alg};    % Current expert model
        % if Pconf>0, calculate CDF
        [Fh,Pconf] = apply_probabilistic_forecaster(cmS,i,cPoints,...
                                         ds_full,ds_event_full,GSet,ConfSet);
                                           
        P_it(alg) = Pconf;     % Confidense of expert alg
        Fh_list{alg} = Fh;     
        %verbose_process = 1;
    end
    
    % Update weights and aggregate experts 
    if sum(P_it)==0      % Error if it is not forecasts
        error('sum(e_active)==0!');
    end
    % Use {сW(1xM) Wconf_row(1xM) Fh_list} to apply numerical integration 
    % Partial CRPS  jf active experts
    eCRPS = nan(1,M+2);  % M exp. + Vovk+ Warmuth
    eLSQ2 = nan(1,M+2);  % M exp. + Vovk+ Warmuth
    % Actual el. loads
    LL_meas = [LL_meas; Yactual];  

    for e=1:M   % For all active
        if P_it(e) > 0 
            Fh = Fh_list{e};  
            eCRPS(e) = ni_crps(Fh.cdf,Fh.xmesh,Yactual);
            eLSQ2(e) = (Fh.yA-Fh.yF)^2;
            PointF(e) = Fh.yF;
            draw_picture = false;
            if draw_picture
                h=figure; 
                [AX,H1,H2] = plotyy(Fh.xmesh,Fh.pdf,Fh.xmesh,Fh.cdf); hold on
                title(['Expert-' num2str(e)]);
                plot([Fh.yA,Fh.yA],[0,1],':r');
                plot([Fh.yF,Fh.yF],[0,1],':g');
                q=1;
                %saveas(h,['dFigures\yy-' num2str(i) '.png']);
                delete(h); 
            end
        end
    end
    LL_pred = [LL_pred; PointF]; 
    P_it_Stored = [P_it_Stored; P_it];
    
    WWwarm_CRPS   = [WWwarm_CRPS; W_it_Warm_CRPS];  % ++ WW by Warm
    WWvovk_CRPS   = [WWvovk_CRPS; W_it_Vovk_CRPS];  % ++ WW by Wovk
    WWetta_LSQ2   = [WWetta_LSQ2; W_it_etta_LSQ2];  % ++ WW by L2
    

    %% Aggregating
    [FaWarm] = aggregate_expert_predictions_by_warmuth(Fh_list,...
                                                    P_it,W_it_Warm_CRPS);
                                                
    ind = find(P_it>0);                                             
    
    [FaVovk] = aggregate_expert_predictions_by_vovk(Fh_list,...
                                                    P_it,W_it_Vovk_CRPS);
       
    eCRPS(M+1) = ni_crps(FaWarm.cdf,FaWarm.xmesh,Yactual);
    eCRPS(M+2) = ni_crps(FaVovk.cdf,FaVovk.xmesh,Yactual);
    aa3D_cdf = [aa3D_cdf,FaVovk.cdf];
    aa3D_pdf = [aa3D_pdf,FaVovk.pdf];
    ST_CRPS = [ST_CRPS;eCRPS];         % Add line
    %% Update weights
    %% W_it_Warm (CRPS)
    ww = W_it_Warm_CRPS;
    for m = 1:M
        if P_it(m)>0
            pw = P_it(m)*eCRPS(m)+(1-P_it(m))*eCRPS(M+1);
        else
            pw = eCRPS(M+1);
        end
        inst_vl_W_CRPS(m) = pw;
        ww(m) = W_it_Warm_CRPS(m)*exp(-0.5*pw/(b-a));
    end
    % Nomalisation
    ww = ww/sum(ww);
    %% Use FixedShare 
    W_it_Warm_CRPS = alpha/M+(1-alpha)*ww;
    
    %% W_it_Vovk (CRPS)
    ww = W_it_Vovk_CRPS;
    for m = 1:M
        if P_it(m)>0
            pw = P_it(m)*eCRPS(m)+(1-P_it(m))*eCRPS(M+2);
        else
            pw = eCRPS(M+2); 
        end
        inst_vl_V_CRPS(m) = pw;
        ww(m) = W_it_Vovk_CRPS(m)*exp(-2*pw/(b-a));
    end
    % Нормируем на 1
    ww = ww/sum(ww);
    %% Use FixedShare 
    W_it_Vovk_CRPS = alpha/M+(1-alpha)*ww;
    
    SVL_W_CRPS = [SVL_W_CRPS;inst_vl_W_CRPS];
    SVL_V_CRPS = [SVL_V_CRPS;inst_vl_V_CRPS];
    q=1;
    
    disp([num2str(rST) ' : ' datestr(DTserial)]); 
    end 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Totals.V=1;
    Totals.W=2;
        
    %  Save results in w_dir
    CM = cummean(ST_CRPS(:,M+1:M+2));  
    save([w_dir '/SummaryTable.mat'],'ST_CRPS','WWwarm_CRPS',...
          'WWvovk_CRPS','LL_pred','LL_meas','SVL_W_CRPS','SVL_V_CRPS',...
                            'ST_LSQ2','WWetta_LSQ2','tikers','Totals',...
                            'SVL_etta_LSQ2','P_it_Stored','GSet','ConfSet');
    CM = cummean(ST_CRPS(:,M+1:M+2)); 
    if GSet.debug
        save([w_dir '/Vovk3dTable.mat'],'variant','aa3D_cdf','aa3D_pdf',...
            'FaVovk','LL_meas','LL_pred','GSet','ConfSet','CM');
    end
    
    disp(['variant ' num2str(variant)]);
    R.variant=variant;
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dt = datestr(now);
    tb = char(9);
    part1 = [dt ' v' num2str(variant) tb ConfSet.train_mode tb ConfSet.aggr_mode];
    part1 = [part1 ConfSet.soft_style '(' num2str(ConfSet.soft_train_min_level) ')'];
    part1 = [part1 '--->Sat,Ngl,M,WA,AA>'];
    part2 = [ConfSet.sat_level ConfSet.neg_level M CM(end,1) CM(end,2)];  
    if GSet.verbose
       [r,str] = txt_write_line('elf_exp_log.txt',part1,part2);
       R.str = str; 
    else
       R.str = '---'; 
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s = ni_crps(f,z,u)
% f - linear approximation of distribution
% z - grid points of argument
% u - test point
% Calculation of CRPS for empirical distribution with compact support
% using integration by trapezoidal method. The integral can be turned 
% into discrete finite sum.
dz = diff(z);  
n=length(z);
s=0;
if u<=z(1)      % (1-F)^2 
    for t=1:n-1
        s=s+dz(t)*((f(t)+f(t+1))^2)*0.25;
    end
elseif u>=z(n)  
    for t=1:n-1
        s=s+dz(t)*(1-(f(t)+f(t+1))/2)^2;
    end
else
    ind = find(z<=u,1,'last'); 
    for t=1:ind-1
        s=s+dz(t)*0.25*(f(t)+f(t+1))^2;
    end
    t=ind;
    d1 = u-z(t);
    d2 = z(t+1)-u;
    d  = z(t+1)-z(t);  
    fu = f(t)+(f(t+1)-f(t))*(d1/(d1+d2));
    fu1= (f(t)+fu)*0.5;
    fu2= (fu+f(t+1))*0.5;
    s = s + d1*(fu1)^2+d2*(1-fu2)^2;
    for t=ind+1:n-1
        s=s+dz(t)*(1-f(t))^2;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [frc_list] = create_expert_list(eMode)
% Create list of experts
% frc - description of expert model (GMM)
% {bag_filter,num_components}
if nargin==0
    eMode = 'LongList';   
end
arg_cell = 'AT_1';
if strcmp(eMode,'LongList')    
    add_spring = 1;
    add_autumn = 1; 
    add_day_periods = 1;   
    add_day_periods_winter = 1;
    add_day_periods_summer = 1;
    add_day_periods_demi = 1;
elseif strcmp(eMode,'ShortList')
    add_spring = 0;
    add_autumn = 0; 
    add_day_periods = 1;   
    add_day_periods_winter = 1;
    add_day_periods_summer = 0;
    add_day_periods_demi   = 0;
else
    error(['Illegal mode value! - ' eMode]);
end

frc_list=cell(0,1); 

%% Any Time 1 (GMM)
frc.method = 'GMM';   % Season GMM Season 
frc.mode = 0;         % 0-Calendar segmentation
frc.predict_mode = 1; % 0-Regression 1-CDF
frc.bag_filter = {'AnyTime'};
frc.comment    = 'AnyTime';
frc.predictors ={arg_cell}';
frc.num_components = 3;
frc.descr = get_frc_description(frc);
frc_list=[frc_list; frc];  % Add to list

%% Seasons %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
frc.method = 'GMM';    % Season GMM Season 
frc.mode = 0;         % 0-Calendar segmentation 
frc.predict_mode = 1; % 0-Regression 1-CDF
frc.bag_filter = {'Winter'};
frc.comment    = 'Winter';
frc.predictors = {arg_cell}';
frc.num_components = 1;
frc.descr = get_frc_description(frc);
frc_list=[frc_list; frc];  %  Add to list

if add_spring
    frc.method = 'GMM';    % Season  
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Spring'};
    frc.comment    = 'Spring';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  %  Add to list
end

frc.method = 'GMM';    % Season  
frc.mode = 0;         % 0-Calendar segmentation 
frc.predict_mode = 1; % 0-Regression 1-CDF
frc.bag_filter = {'Summer'};
frc.comment    = 'Summer';
frc.predictors ={arg_cell}';
frc.num_components = 1;
frc.descr = get_frc_description(frc);
frc_list=[frc_list; frc];  

if add_autumn
    frc.method = 'GMM';   % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Autumn'};
    frc.comment    = 'Autumn';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc]; 
end
   

%% Winter + %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if add_day_periods_winter
    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Winter','Morning'};
    frc.comment    = 'Winter+Morning';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  

    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Winter','DayTime'};
    frc.comment    = 'Winter+DayTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  

    frc.method = 'GMM';   % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Winter','Evening'};
    frc.comment    = 'Winter+Evening';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  

    frc.method = 'GMM';   % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Winter','NightTime'};
    frc.comment    = 'Winter+NightTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  
end

if add_day_periods_demi 
    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Spring','Morning'};
    frc.comment    = 'Spring+Morning';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  

    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation 
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Spring','DayTime'};
    frc.comment    = 'Spring+DayTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  

    frc.method = 'GMM'; 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Spring','Evening'};
    frc.comment    = 'Spring+Evening';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  
    
    frc.method = 'GMM'; 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Spring','NightTime'};
    frc.comment    = 'Spring+NightTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];  
end

if add_day_periods_summer
    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Summer','Morning'};
    frc.comment    = 'Summer+Morning';
    frc.predictors ={arg_cell}';
    frc.descr = get_frc_description(frc);
    frc.num_components = 1;
    frc_list=[frc_list; frc];  
    
    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Summer','DayTime'};
    frc.comment    = 'Summer+DayTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 1;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];

    frc.method = 'GMM'; 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Summer','Evening'};
    frc.comment    = 'Summer+Evening';
    frc.predictors ={arg_cell}';
    frc.num_components = 1;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];

    frc.method = 'GMM'; 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Summer','NightTime'};
    frc.comment    = 'Summer+NightTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 1;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc]; 
end

if add_day_periods_demi
    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Autumn','Morning'};
    frc.comment    = 'Autumn+Morning';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc]; 
    
    frc.method = 'GMM';    % Season GMM Season 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Autumn','DayTime'};
    frc.comment    = 'Autumn+DayTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc];

    frc.method = 'GMM'; 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Autumn','Evening'};
    frc.comment    = 'Autumn+Evening';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc]; 

    frc.method = 'GMM'; 
    frc.mode = 0;         % 0-Calendar segmentation
    frc.predict_mode = 1; % 0-Regression 1-CDF
    frc.bag_filter = {'Autumn','NightTime'};
    frc.comment    = 'Autumn+NightTime';
    frc.predictors ={arg_cell}';
    frc.num_components = 2;
    frc.descr = get_frc_description(frc);
    frc_list=[frc_list; frc]; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(frc_list)
    frc_list{i}.ind = i;
    v_list = get_var_list(frc_list{i}.predictors);   
end
q=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function descr = get_frc_description(frc)
s=[frc.bag_filter{1}];
if length(frc.bag_filter)>1
    for i=2:length(frc.bag_filter)
        s=[s '+' frc.bag_filter{i}];
    end
end
s = [frc.method '/' num2str(frc.num_components)  '/' s];
descr = s;


function [v_list] = get_var_list(list)
v_list = ['_Vars: ' list{1}]; 
if length(list)>1 
    for i=2:length(list)
        v_list = [v_list ',' list{i}];
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Totals] = draw_elf_results(tag)
% To create FIG and EPS figures, set paper_figures to true!
% Draw results of explore_elf14_data 
% Load data from {variant1, variant2, variant3} folders
% Save drawings in Figures folder
paper_figures = false;

if nargin==0
    tag='21';
end
figure_path = '..\figures2\';
draw_legend  = 0;
verbose_all  = 0;
save_figures = 0;
log_file_name = 'draw_elf14_results_log.txt';
folder_list = {'variant1', 'variant2', 'variant3', 'variant4'};

StList = cell(4,1)
%% Collecting data of 3 variants to display results
AggrWVLosses = [];   % 
for v = 2:4
    w_dir = folder_list{v};
    fname = [w_dir '/SummaryTable.mat'];
    load(fname);    %
    %% Sample size and number of experts
    S = [];      % Structure
    [T,Mpa] = size(ST_CRPS); % Instant CRPS of experts and 21 aggregators
    M = Mpa-2;   % M+1->Warmuth; M+2>Vovk 
    [T,N] = size(WWvovk_CRPS);
    S.expLossses = ST_CRPS(:,1:M);
    S.wvLossses = ST_CRPS(:,M+1:M+2);
    S.P_it_Stored = P_it_Stored;
    AggrWVLosses = [AggrWVLosses S.wvLossses];   
    StList{v} = S;
end

% Matrix of expert losses 
S = StList{4}; % UNIFORM Aggregation
UNI_CRPS = S.expLossses;     % Just experts
%  AnyTimeExpLosses
Any_Time  = UNI_CRPS(:,1);   %  AnyTime only
% Just 20 experts  All_but1
All_but_1 = UNI_CRPS(:,2:M); % All but AnyTime 
% W+V (Soft, Hard, Uniform) AggrWVLosses
% AggrWVLosses
%%
% 'CumLosses of Experts and Aggregators');
h = figure;
plot(cumsum(All_but_1),'LineWidth',0.5);  hold on 
plot(cumsum(Any_Time),'--k','LineWidth',2);  hold on 
plot(cumsum(AggrWVLosses),'LineWidth',1);
q=1;
xlim([1 T]);
ylim([0 100000]);
grid on
if paper_figures
    saveas(h,[figure_path 'Fig9l.fig']);
    saveas(gcf,[figure_path 'Fig9l'],'epsc2');
end
saveas(h,[figure_path 'Fig9l.png']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 'Cumulated Means of Losses (Experts and Aggregators);
h = figure;
plot(cummean(All_but_1),'LineWidth',0.5);  hold on 
%plot(cummean(Any_Time),'LineWidth',2);  hold on 
plot(cummean(Any_Time),'--k','LineWidth',2);  hold on 
plot(cummean(AggrWVLosses),'LineWidth',1);
q=1;
xlim([3 T]);
ylim([6 20]);
grid on
if paper_figures
    saveas(h,[figure_path 'Fig9r.fig']);
    saveas(gcf,[figure_path 'Fig9r'],'epsc2');
end
saveas(h,[figure_path 'Fig9r.png']);

%% Regret Trajectories for AA and  WA 
if true
    S = StList{2};
    L_it = S.expLossses;
    L_wv = S.wvLossses; 
    P_it_Stored = S.P_it_Stored;
    for t=1:T    
        ind = isnan(L_it(t,:));
        L_it(t,ind) = 0;
    end
    etaVovk = 2/(GSet.b-GSet.a);   C_Vovk = log(N)/etaVovk;
    etaWarm = 0.5/(GSet.b-GSet.a); C_Warm = log(N)/etaWarm;
    hVovk = repmat(L_wv(:,2),1,N);
    hWarm = repmat(L_wv(:,1),1,N);
    hVovk_Lit = hVovk-L_it;
    hWarm_Lit = hWarm-L_it;
    for j = 1:N
        Rvovk(:,j) = hVovk_Lit(:,j) .* P_it_Stored(:,j);
        Rwarm(:,j) = hWarm_Lit(:,j) .* P_it_Stored(:,j);
    end    
    h7 = figure;
    subplot(1,2,1);
    plot(cumsum(Rvovk)); hold on
    plot([1 T],[C_Vovk C_Vovk],':r','LineWidth',2);
    ylim([-3000 3100]);
    xlim([0 T+1]);
    subplot(1,2,2);
    plot(cumsum(Rwarm)); hold on
    plot([1 T],[C_Warm C_Warm],':b','LineWidth',2);
    ylim([-3000 3100]);
    xlim([0 T+1]);
    if paper_figures
        saveas(gcf,[figure_path 'fig7.fig']);
        saveas(gcf,[figure_path 'fig7'],'epsc2');
    end
    saveas(gcf,[figure_path 'fig7.png']);
end


function  [h] = show_prior_confidence(frc_list,events,GSet,ConfSet)  
% Area plot of confidence 
    M = length(frc_list);
    PriorComp = nan(size(events,1),M);
    for j=1:M
        frc = frc_list{j};
        [comp_z] = evaluate_bag_filter(frc.bag_filter,events,GSet,ConfSet);
        %comp_z = comp_z.^ConfSet.z_power;   %%!!!!!!!
        PriorComp(:,j) = comp_z; 
    end    
    h = figure;
    sPC = PriorComp;
    sm = sum(sPC');
    for i = 1:length(comp_z)
        sPC(i,:) = sPC(i,:)/sm(i);
    end
    area(sPC)
    ylim([0 1])
    xlim([0 length(comp_z)]);
    title('Normalised Prior Expert Confidence') 
    q=1;

function r = get_mean_rss(d)   % sum(d(i)^2)
r=((d(:)') * d(:)) / length(d(:));

function [mStruct,ind,confidence] = ...
         train_ds_forecaster(frc,ds,ds_event,varY,GSet,ConfSet)
% frc.method : {'RF','GLM',...}
% frc.bag_filter
mStruct.method = frc.method;  
mStruct.mode   = frc.mode;    
mStruct.model  = [];          
mStruct.bag_filter = frc.bag_filter; 
mStruct.comment = frc.comment;
mStruct.num_components = frc.num_components;
mStruct.ind = frc.ind;
% Create confidence 
ConfSet.action = 'train';
[comp_z] = evaluate_bag_filter(frc.bag_filter,ds_event,GSet,ConfSet);
confidence = comp_z;

if strcmp(ConfSet.train_mode,'SOFT')
    if strcmp(ConfSet.soft_style,'ByLevel')
        ind = find(comp_z>=ConfSet.soft_train_min_level);
        ds = ds(ind,:);
    elseif strcmp(ConfSet.soft_style,'Random')
        rV =rand(length(comp_z),1);
        ind = find(comp_z>=rV);
        ds = ds(ind,:);
    else
        error('Illegal soft_style !');    
    end
elseif strcmp(ConfSet.train_mode,'HARD') 
    ind = find(comp_z>= ConfSet.hard_train_min_level);
    ds = ds(ind,:);
elseif strcmp(ConfSet.train_mode,'WIDE')
    ind = find(comp_z> 0);
    ds = ds(ind,:);
else
    error('Illegal ConfSet.train_mode !');
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


txt_part = ['V. ' num2str(GSet.variant) '-' ConfSet.train_mode '-' ];
txt_part = [txt_part ConfSet.soft_style '-' mStruct.comment];
if GSet.verbose
    [r,str] = txt_write_line('elf_train_log.txt',txt_part,...
                                               [mStruct.ind  size(ds)]);
end                                           
mStruct.varX = frc.predictors; 
mStruct.varY = varY;           
mStruct.predict_mode = frc.predict_mode; % 0-Regression 1-CDF

Y = table2array(ds(:,varY)); 

if strcmp(mStruct.method,'GMM')    
    mStruct.model = create_gmm_model_msk(ds,mStruct,GSet,ConfSet)
    Y_h = predict_gmm_mean(mStruct,ds,GSet);                        %%%%%%
    mStruct.RSStrn = get_mean_rss(Y-Y_h);                           %%%%%%
    mStruct.TrainSize=length(Y);
else   
    mStruct.model = [];
    error(['Illegal model type :' frc.method]);
end

Y_h_nan = isnan(Y_h);
if sum(Y_h_nan)>0
    beep;
    disp('Nan in train_ds_forecasts (Y_h)!');
    return;
end

function [comp_z] = evaluate_bag_filter(bag_filter,ds_event,GSet,ConfSet)
[n,d] = size(ds_event);
comp_z = ones(n,1);
for i=1:length(bag_filter)
   %comp_z = comp_z .* double(ds_event(:,bag_filter{i}));
   comp_z = comp_z .* table2array(ds_event(:,bag_filter{i}));
end

if ConfSet.neg_level>0
    ind =  find(comp_z < ConfSet.neg_level);
    comp_z(ind)=0;
end

if ConfSet.sat_level<1 
    ind = find(comp_z >= ConfSet.sat_level);  
    comp_z(ind)=ConfSet.sat_level;            
    alpha = 1/ConfSet.sat_level;              
    comp_z = comp_z*alpha;
end
if ~ConfSet.z_power==1 
    comp_z = comp_z.^ConfSet.z_power;
end


function [Fa] = aggregate_expert_predictions_by_warmuth(F_list,...
                                                        p_it,w_it_Warm)
Fa = [];

w_it = w_it_Warm .* p_it;
w_it = w_it/sum(w_it);  % Текущие веса по Вармуту 
ind = find(w_it>0);

for j = 1:length(ind)
    k = ind(j);
    if j == 1
        Fa.cdf   = F_list{k}.cdf*w_it(k);         
        Fa.xmesh = F_list{k}.xmesh;               
    else
        Fa.cdf = Fa.cdf + F_list{k}.cdf*w_it(k);
    end
end

Fa.pdf = [diff(Fa.cdf); 0];
Fa.pdf = Fa.pdf/sum(Fa.pdf);

verbose=0;
if verbose
   figure;
   for k=1:length(ind)
       f = F_list{ind(k)};
       plot(f.xmesh,f.cdf); hold on
   end
   plot(Fa.xmesh,Fa.cdf,':b','LineWidth',2);
   hold off; 
      grid on;
   title('Aggregating by Warmuth');
end

function [Fa] = aggregate_expert_predictions_by_vovk(F_list,...
                                                     p_it,W_it_Vovk)
w_it = W_it_Vovk .* p_it;              
w_it = w_it/sum(w_it);                 
ind = find(w_it>0);                    
if length(ind)==1
    q=1;
end
F1 = F_list{ind(1)};                   % As a template
Fa.xmesh = F1.xmesh;  
z  = Fa.xmesh;                         % Support
A = zeros(length(ind),length(z));      % Nomi (nator)
B = zeros(length(ind),length(z));      % DENO (minator)
for j = 1:length(ind)                  % Active experts  
    k = ind(j);                        % Position
    ff = (F_list{k}.cdf)' .^2;         % cdf^2
    A(j,:) = exp(-2*ff);               % 
    %gg = (1-ff).^2;               
    gg = (1-F_list{k}.cdf)' .^2;
    B(j,:) = exp(-2*gg);
end
w = w_it(ind);                         % Weihts      
deno = w*B;                            
nume = w*A;

% Sum... 
Fa.cdf = (0.5-0.25*log(nume ./ deno))';
%Fa.cdf = (0.5-0.25*log(R'))';
%Fa.cdf = (0.5-0.25*log(R));
Fa.pdf = [diff(Fa.cdf); 0];
Fa.pdf = Fa.pdf/sum(Fa.pdf);          % Norm  

verbose=0;
if verbose
   figure;
   for k=1:length(ind)
       f = F_list{ind(k)};
       plot(f.xmesh,f.cdf); hold on
   end
   plot(Fa.xmesh,Fa.cdf,':b','LineWidth',2);
   hold off;
   grid on;
   title('Aggregating by Vovk');
end

function [Yh,comp_z] = apply_ds_forecaster(mS,ds,ds_event,MM,GSet,ConfSet)
if mS.mode==0
    [comp_z] = evaluate_bag_filter(mS.bag_filter,ds_event,GSet,ConfSet);
else
    error('Illegal mS.mode in apply_ds_forecaster!');
    [comp_z] = MM(:,mS.mode);
end;
if strcmp(ConfSet.action,'train') % remove <0.99 
    if strcmp(ConfSet.train_mode,'HARD')
        ds = ds(comp_z>0.99,:);
    end
end

Yh = []; % Вектор-столбец точечные оценки

if strcmp(mS.method,'RF')
    X=table2array(ds(:,mS.varX));
    Yh = predict(mS.model, X);
elseif strcmp(mS.method,'GLM')
    Yh = predict(mS.model, ds);
elseif strcmp(mS.method,'LR')    
    %X=table2array(ds(:,mS.varX));
    Yh = predict(mS.model,ds);
    %% Place to create ensemble
elseif strcmp(mS.method,'GMM')         
    %X=table2array(ds(:,mS.varX));     
    Yh = predict_gmm_mean(mS,ds,GSet);
else
    error(['Apply_ds_forecaster - Not implemented yet for '...
            mS.method]);
end

a=isnan(Yh);
if sum(a)>0
    beep;
    error('Nans in Yh!');
end

%% Возвращает вероятностный прогноз отклика модели mS для i-ой точки ds_full
function [Fh,Wconf] = apply_probabilistic_forecaster(mS,i,cPoints,...
                                               ds_full,ds_event_full,GSet,ConfSet)
% INPUT:                                       
% mS                       -  current model (expert)
% i                        -  time point
Wconf = evaluate_bag_filter(mS.bag_filter,ds_event_full(i,:),GSet,ConfSet);

if strcmp(ConfSet.aggr_mode,'Soft') 
    % Nothing to do - use Wconf as is
elseif strcmp(ConfSet.aggr_mode,'Hard')  % Hard mode (Expert is Active!)
    if Wconf > 0.99                   % Hard selection of train samples
        Wconf=1;   %% Is Active
    else
        Wconf=0;   %% Sleeping! Is not active  
    end
end

if Wconf==0     
    Wconf= ConfSet.min_conf_level;
end

%if Wconf==0    
    
if ConfSet.uniform_conf_level % !! Дублирование "uniform"  !!
    Wconf = 1;
end

Yactual = table2array(ds_full(i,mS.varY)); % Проверочное значение

if strcmp(mS.method,'GMM')
    d = (GSet.b-GSet.a)/1000;
    pts = (GSet.a: d :GSet.b);    % Сетка эл. нагрузок
    Xactual = table2array(ds_full(i,mS.varX));   
    pdf_raw = pdf(mS.model,[Xactual*ones(length(pts),1),pts']);
    pdf_raw_sum = sum(pdf_raw);
    pdf_cor = pdf_raw/pdf_raw_sum;
    if false
        figure;
        plot([pdf_raw pdf_cor]);
        q=1;
    end
    Fh.pdf = pdf_cor;
    Fh.cdf = cumsum(pdf_cor);
    Fh.xmesh = pts'; 
    ExpLoad = (Fh.xmesh')*Fh.pdf;  % Оценка мат. ожидания yh
    Fh.yA = Yactual;
    Fh.yF = ExpLoad;
           
    show_results = 0;
    if show_results
        figure;
        plot(Fh.xmesh,Fh.cdf,'-g','LineWidth',2);
        title([mS.comment ': Estimation of conditional distribution']);
    end
    
    create_picture = 0;
    if create_picture
        h=figure; 
        [AX,H1,H2] = plotyy(Fh.xmesh,Fh.pdf,Fh.xmesh,Fh.cdf); hold on
        plot([ExpLoad,ExpLoad],[0,1],':b');
        title(rcd([num2str(i) '-' mS.comment]));
        plot([Yactual,Yactual],[0,1],':r');
        q=1;
        if save_pictures
            saveas(h,rcd(['dFigures\e-' mS.comment '-' num2str(i) '.png']));
        end
        delete(h); 
    end
else
    error('Illegal mS.method!');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [r] = rcd(file_name)
    r = strrep(file_name,'_','-');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function   GMModel = create_gmm_model_msk(ds,mStruct,GSet,ConfSet)
% Get K of mStruct
    % Scatter Temperature+Load 
    if GSet.UseAveTemp
        X      = table2array([ds(:,'AveTemp') ds(:,mStruct.varY)]);
    else
        X      = table2array([ds(:,mStruct.varX) ds(:,mStruct.varY)]);
    end
    [n,d]  = size(X);
    
    % Save train sample for expert mStruct.ind
    if GSet.debug
        save(['Train_Samples\V' num2str(GSet.variant) '\ExpTrainSample_' num2str(mStruct.ind) '.mat'],...
        'X','n','mStruct');
    end
    
    [X_s,ind_s] = sort(X(:,1));  
    
    K = mStruct.num_components; 
    %% K = 2; 
    % Divide on K parts and set labels for EM start       
    tt = n/K;
    for i = 1:n
        j = ind_s(i);
        g(j) = fix((i-1)/tt)+1;
    end
                     
    delta = 0.0001;
    GMModel = fitgmdist(X,K,'RegularizationValue',delta,...
                       'Start',g,'Options',statset('MaxIter',200));
    q=1;
    
    if GSet.verbose
        P = posterior(GMModel,X); 
        [np,dp] = size(P);
        G = ones(np,1); 
        for i = 1:np
            ppi = P(i,:);
            G(i) = get_coin(ppi);
        end
        DetailedFigure = 0;
        if DetailedFigure
            h1 = figure('Name',[mStruct.comment],'Position',[400 100 900 500]);
            subplot(1,4,1);     % Raw data
            G0 = ones(np,1);
            hx1 = gscatter(X(:,1),X(:,2),G0,'bgrcm','.',2,'off'); hold on;
            xlim([GSet.T_min GSet.T_max]);
            ylim([GSet.a GSet.b]);
            xlabel('Temperature');
            ylabel('Electrical Load');
            title('Original');
            
            subplot(1,4,2);     % Sort by temperature
            hx2 = gscatter(X(:,1),X(:,2),g,'bgrcm','.',2,'off'); hold on;
            xlim([GSet.T_min GSet.T_max]);
            ylim([GSet.a GSet.b]);
            xlabel('Temperature');
            ylabel('Electrical Load');
            title('Colored by T');
            subplot(1,4,3);     
            hx3 = gscatter(X(:,1),X(:,2),G,'bgrcm','.',2,'off'); hold on;
            xlabel('Temperature');
            ylabel('Electrical Load');
            title('Fitted by GMDIST');
            xlim([GSet.T_min GSet.T_max]);
            ylim([GSet.a GSet.b]);
            grid on
        % Centers and ellipses
%         hx2 = gscatter(GMModel.mu(:,1),GMModel.mu(:,2),[1 2]','br',...
%             '+o',16,'off');
%         set(hx2,'LineWidth',3); 
            subplot(1,4,4);     % GMM synthesized data
            [Z,Gsynt] = random(GMModel,np);
            hx2 = gscatter(Z(:,1),Z(:,2),Gsynt,'bgrcm','.',2,'off'); hold on;
            xlabel('Temperature');
            ylabel('Electrical Load');
            title('Synthesized');
            xlim([GSet.T_min GSet.T_max]);
            ylim([GSet.a GSet.b]);
            grid on
            saveas(h1,['illustrations\' mStruct.comment '.png']);
            delete(h1);
        else % Just train sample(grouped)
             h1 = figure('Name',[mStruct.comment]);
             % Actual train sample(grouped)
             hx1 = gscatter(X(:,1),X(:,2),G,'bgrcm','.',2,'off'); hold on;
             xlabel('Temperature');
             ylabel('Electrical Load');
             title(['Expert: ' mStruct.comment]);
             xlim([GSet.T_min GSet.T_max]);
             %ylim([GSet.a GSet.b]);
             ylim([50 350]);
             grid on
             if GSet.debug
                saveas(h1,['illustrations\' mStruct.comment '.png']);
             end
             delete(h1);
        end
    end

function   Y_h = predict_gmm_mean(mS,ds,GSet)
%% Calculation of means 
L_grid = GSet.L_grid';  
T_grid = GSet.T_grid;  
gmmModel = mS.model;
X = table2array(ds(:,mS.varX)); 

h = histogram(X);
mm = zeros(h.NumBins,1);
for b = 1:h.NumBins
    xC = (h.BinEdges(b)+h.BinEdges(b+1))/2;
    pB = pdf(gmmModel,[ones(length(L_grid),1)*xC L_grid]);
    pB = pB/sum(pB);
    mm(b) = L_grid'*pB;
end
ind = discretize(X,h.BinEdges);
Y_h = mm(ind);
delete(h);



function r = get_coin(pt)
pt = pt(:);
d=rand;
w=cumsum(pt);
r=find(d<=w,1,'first');

function [r,str] = txt_write_line(file_name,name_str,values)
v=values(:);
k=length(v);
tb=char(9);
lf=sprintf('\n');
rfid=fopen(file_name,'a+'); 
ss=name_str;
for i=1:k
    s=num2str(v(i));
    ss=[ss tb s];
end;

ss=[ss lf];
fprintf(rfid,ss);
fclose(rfid);

if nargout>0
    str=ss;
else
    disp(ss);
end;
r=1;

function [cmS,confid] = get_sample_conf(frc,ds_event,GSet,ConfSet)
ConfSet.action = 'test';
[comp_z] = evaluate_bag_filter(frc.bag_filter,ds_event,GSet,ConfSet);
confid = comp_z;
cmS = frc.bag_filter;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                      




