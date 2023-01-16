function varargout=sparse_distanceXY(X,Y,varargin)

% based on X, calculates the distance between each X and the closest Y
% AFTER each X

tolerance=1e-4;
if nargin>2
    while length(varargin)>1
        switch varargin{1}
            case 'tolerance'
                tolerance=varargin{2};
        end
        varargin(1:2)=[];
    end
end

    

if ~issorted(X) | ~issorted(Y)
    error('inputs not sorted');
end

if (isempty(X) | isempty(Y))
    dist=[];
    Yout=[];
    switch nargout
        case 1
            varargout{1}=dist;
        case 2
            varargout{1}=dist;
            varargout{2}=Yout;
        case 3
            varargout{1}=dist;
            varargout{2}=Yout;
            varargout{3}=[];
    end

else

    X=X(:)';
    Y=Y(:)';   % deal with row vectors
    
    X=double(X);
    Y=double(Y);
    
    Xplus=X+tolerance;

    [sorted, ind]=sort([Xplus,Y]);
    ind=find(ind<=length(X));
    Yind=ind(1:length(X))-[0:1:(length(X)-1)];
    i=find(Yind>length(Y));
    Yind(i)=length(Y);
    dist=Y(Yind)-X;
    dist(i)=Inf;
    Yout=Y(Yind);
    Yout(i)=Inf;
    switch nargout
        case 1
            varargout{1}=dist;
        case 2
            varargout{1}=dist;
            varargout{2}=Yout;
        case 3
            varargout{1}=dist;
            varargout{2}=Yout;
            varargout{3}=Yind;
    end
end

