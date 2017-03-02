%% This is using forward model for maxent by 17th Feb 2017
% Signal is generated using this forward model, and ideal pixelated image
% Last edit: 21-02-17
% Pragya Sharma
function tv_fwdmod_nosig()
    %% Generating A matrix
    c = physconst('LightSpeed');
    f = 'f3';
    Ntag = 16; 
    Nrecv = 4;
    RCS = 0.5;
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
    Nfreq = length(Freq);
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
            Pos_tag = [500 -300;500 -100; 500 100;500 300;300 500;100 500;-100 500;...
               -300 500;-500 300;-500 100;-500 -100;-500 -300;-300 -500; ...
               -100 -500;100, -500;300 -500].*0.001;
           Ntag = 16;
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
        otherwise
            disp('Error in Pos_recv');
    end
    figure;
    plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
    hold on;
    plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);

    % For following expression of e, tag/ receiver cannot be on some grid
    % point. R1 or R2 cannot be zero
    x_v = linspace(-0.45,0.45,151); % x at corner of pixel
    y_v = x_v;

    lx = length(x_v);
    [X, Y] = meshgrid(x_v,y_v);

    P = combvec(x_v,y_v)';

    R1 = repmat(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Ntag 1]) ...
    - repmat(permute(Pos_tag, [3 1 2]), [lx*lx 1 1]) ).^2, 3)),1,Nrecv);

    dummy1 = ones(1,Ntag);
    R2 = kron(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Nrecv 1]) ...
    - repmat(permute(Pos_recv, [3 1 2]), [lx*lx 1 1]) ).^2, 3)), dummy1);

    R1 = repmat(R1,1,Nfreq); % All tag, recv, freq
    R2 = repmat(R2,1,Nfreq); % All tag, recv, freq
    dummy3 = ones(1,Ntag*Nrecv);
    Freq = repmat(kron(Freq,dummy3),[lx*lx 1]);
    % The amplitude is changed
    e1 = exp(-1j*(2*pi/c)*(Freq.*(R1+R2)));
    e_fwd = sqrt(RCS/(4*pi))*(1/(4*pi))*(c./(Freq.*R1.*R2)).*e1; 
    e_fwd = transpose(e_fwd); % == A matrix
    e = e_fwd;
    [l,~] = size(e_fwd);
    assignin('base','e',e);

    %% Generating V (== y) matrix
    img = zeros(lx,lx);
    img(69:75,17:23) = 1;
    figure;
    mesh(X,Y,img'); colormap('gray')
    x = reshape(img,[],1);
    std_dev = 0.05;
    e_bwd = transpose(e1);
    V = e_fwd*x + std_dev*randn(l,1);
    B = e_bwd'*V; B = abs(B);
    B = (reshape(B, [lx lx]));
    figure;
    mesh(X,Y,B')
    axis 'square'
end
