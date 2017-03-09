%% This is using forward model by 17th Feb 2017
% Signal is generated using this forward model, and ideal pixelated image
% Last edit: 02-03-17 - "Introduced Randomness" - not needed.
% Pragya Sharma
function cs_fwdmod_nosig()
    %% Generating A matrix
    c = physconst('LightSpeed');
    f = 'f5';
    Ntag = 16; 
    Nrecv = 4;
    RCS = 0.1;
    flag_rand = 0; % 1 == random only for frequency
    rng(3,'twister');
    obj = 'sparse'; % block or sparse
    switch(f)     
        case 'f3'
            Freq =(1.7:0.05:2.2)*1e9;               
        case 'f5'
            Freq = [1.79 2.03 2.2]*1e9; % 0.9 1.5 2.2
        case 'f6'
            Freq = 2e9;
        otherwise
            display('Error in frequency');
    end
    if (flag_rand)
        Nfreq_temp = length(Freq);
        Nfreq = 11;
        q = randperm(Nfreq_temp);
        Freq = Freq(q(1:Nfreq));
    else
        Nfreq = length(Freq);
    end
    switch(Ntag)
        case 32
            Pos_tag = zeros(Ntag,2);
            for i=1:Ntag/4
                Pos_tag(i, :)=[-0.35+(i-1)*0.1 -0.5] ;
            end
            for i=1:Ntag/4
                Pos_tag(i+(Ntag/4),:)=[0.5 -0.35+(i-1)*0.1] ;
            end
            for i=1:Ntag/4
                Pos_tag(i+(2*Ntag/4), :)=[-0.35+(i-1)*0.1 0.5] ;
            end
            for i=1:Ntag/4
                Pos_tag(i+(3*Ntag/4), :)=[-0.5 -0.35+(i-1)*0.1] ;
            end
        case 16  
            Pos_tag = zeros(Ntag,2);
            for i=1:Ntag/4
                Pos_tag(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
            end
            for i=1:Ntag/4
                Pos_tag(i+(Ntag/4),:)=[0.5 -0.3+(i-1)*0.2] ;
            end
            for i=1:Ntag/4
                Pos_tag(i+(2*Ntag/4), :)=[-0.3+(i-1)*0.2 0.5] ;
            end
            for i=1:Ntag/4
                Pos_tag(i+(3*Ntag/4), :)=[-0.5 -0.3+(i-1)*0.2] ;
            end
        case 4
            Pos_tag = [-0.5,0.5;-0.1667,0.5;0.1667,0.5;0.5,0.5];
        case 'cst'
            Ntag = 16;
            Pos_tag = [500 -300;500 -100; 500 100;500 300;300 500;100 500;-100 500;...
               -300 500;-500 300;-500 100;-500 -100;-500 -300;-300 -500; ...
               -100 -500;100, -500;300 -500].*0.001;           
        case 'rand'
            Ntag = 16;
            Pos_tag = [(0.5-(-0.5)).*rand(Ntag/4,1) - 0.5, - 0.5*ones(Ntag/4,1);
                0.5*ones(Ntag/4,1), (0.5-(-0.5)).*rand(Ntag/4,1) - 0.5;
                (0.5-(-0.5)).*rand(Ntag/4,1) - 0.5, 0.5*ones(Ntag/4,1);
                -0.5*ones(Ntag/4,1), (0.5-(-0.5)).*rand(Ntag/4,1) - 0.5];

        otherwise
            disp('Error in Pos_tag');
    end
    switch(Nrecv)
        case 1
            Pos_recv = [-0.5,-0.5];
        case 2
            Pos_recv = [-0.5,-0.5;0.5,-0.5];   
        case 4
            Pos_recv = [-0.5,-0.5;0.5,-0.5;0.5,0.5;-0.5,0.5];
        case 6
            Pos_recv = zeros(Nrecv,2);
            Pos_recv(:,1) = linspace(-0.5,0.5,6);
            Pos_recv(:,2) = ones(Nrecv,1).*-0.5;
        case 16
            Pos_recv = zeros(Nrecv,2);
            for i=1:Nrecv/4
                Pos_recv(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
            end
            for i=1:Nrecv/4
                Pos_recv(i+(Nrecv/4),:)=[0.5 -0.3+(i-1)*0.2] ;
            end
            for i=1:Nrecv/4
                Pos_recv(i+(2*Nrecv/4), :)=[-0.3+(i-1)*0.2 0.5] ;
            end
            for i=1:Nrecv/4
                Pos_recv(i+(3*Nrecv/4), :)=[-0.5 -0.3+(i-1)*0.2] ;
            end
        case 'cst'
            Pos_recv = [500 -500; 500 500; -500 500; -500 -500].*0.001;
            Nrecv = 4;
        case 'rand'
            Nrecv = 4;
            Pos_recv = [(0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5, - 0.5*ones(Nrecv/4,1);
                0.5*ones(Nrecv/4,1), (0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5;
                (0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5, 0.5*ones(Nrecv/4,1);
                -0.5*ones(Nrecv/4,1), (0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5];
        otherwise
            disp('Error in Pos_recv');
    end
    figure;
    plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
    hold on;
    plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);

    % For following expression of e, tag/ receiver cannot be on some grid
    % point. R1 or R2 cannot be zero
    Ngridx = 80;
    x_v = linspace(-0.4,0.4,Ngridx); % x at corner of pixel
    y_v = x_v;

    [X, Y] = meshgrid(x_v,y_v);

    P = combvec(x_v,y_v)';

    R1 = repmat(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Ntag 1]) ...
    - repmat(permute(Pos_tag, [3 1 2]), [Ngridx*Ngridx 1 1]) ).^2, 3)),1,Nrecv);

    dummy1 = ones(1,Ntag);
    R2 = kron(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Nrecv 1]) ...
    - repmat(permute(Pos_recv, [3 1 2]), [Ngridx*Ngridx 1 1]) ).^2, 3)), dummy1);

    R1 = repmat(R1,1,Nfreq); % All tag, recv, freq
    R2 = repmat(R2,1,Nfreq); % All tag, recv, freq
    dummy3 = ones(1,Ntag*Nrecv);
    Freq = repmat(kron(Freq,dummy3),[Ngridx*Ngridx 1]);
    % The amplitude is changed
    e1 = exp(-1j*(2*pi/c)*(Freq.*(R1+R2)));
    e_fwd = sqrt(RCS/(4*pi))*(1/(4*pi))*(c./(Freq.*R1.*R2)).*e1; 
    e_fwd = transpose(e_fwd); % == A matrix
    e = e_fwd;
    [l,~] = size(e);
    assignin('base','e',e);

    %% Generating V (== y) matrix
    img = zeros(Ngridx,Ngridx);
    switch obj
        case 'sparse'
            Nscat = 6;
            grid_index = 1:Ngridx;
            temp2 = randperm(Ngridx);
            xpos = grid_index(temp2(1:Nscat));
            temp3 = randperm(Ngridx);
            ypos = grid_index(temp3(1:Nscat));
            for i = 1:Nscat
                img(xpos(i),ypos(i)) = 1;
            end
        case 'block'
    %         img(39:42,17:23) = 1;
            img(49:52,17:22) = 1;
        otherwise
            fprintf('This object doesn''t exist.\n');
    end
    figure;
    mesh(X,Y,(abs(img))'); colormap('gray'); axis 'square'; view(0,90)
    axis([-0.5,0.5,-0.5,0.5]);
    xlabel('x-axis','FontSize',14);
    ylabel('y-axis','FontSize',14); colormap('cool');
    x = reshape(img,[],1);
    std_dev = 0.0;
    e_bwd = transpose(e1);
    V = e*x;
    rand_noise = std_dev*randn(l,1).*V;
    V = V + rand_noise;
    assignin('base','V',V);
    B = e_bwd'*V; B = abs(B);
    B = (reshape(B, [Ngridx Ngridx]));
    figure;
    mesh(X,Y,B'); view(0,90)
    axis 'square'; 
    axis([-0.5,0.5,-0.5,0.5]);
    xlabel('x-axis','FontSize',14);
    ylabel('y-axis','FontSize',14);

end
