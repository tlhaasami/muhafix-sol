// Demo contract written for Muhafiz.sol: the same reentrancy mistake as the
// bank example, hidden behind unusual names.
pragma solidity ^0.4.24;

contract Piggy {
    mapping(address => uint256) public stash;

    function put() public payable {
        stash[msg.sender] += msg.value;
    }

    function grab(uint256 q) public {
        if (stash[msg.sender] >= q) {
            if (msg.sender.call.value(q)()) {
                stash[msg.sender] -= q;
            }
        }
    }
}
