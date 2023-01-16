function bin=hexmat2bin(HexMat)
  % returns cell array of binary bits in input hex array of characters
  % output cell array is same shape as input char() array
  %
  % hexarray=['1','2','0';'A','C','8';'9','6','F'];
  % bincell=hexmat2bin(hexarray)
  %  bincell =
  %
  %  {'0001'}    {'0010'}    {'0000'}
  %  {'1010'}    {'1100'}    {'1000'}
  %  {'1001'}    {'0110'}    {'1111'}
    [m,n]=size(HexMat);
    HexC=mat2cell(HexMat,ones(1,m),ones(1,n));
    bin=reshape(cellstr(dec2bin(hex2dec(HexC),4)),m,[]);
  end