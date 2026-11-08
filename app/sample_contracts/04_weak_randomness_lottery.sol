// Demo contract written for Muhafiz.sol: a lottery whose "random" winner
// is computed from block data that a miner can see or influence.
pragma solidity ^0.4.24;

contract BlockLottery {
    address[] public players;
    uint256 public ticketPrice = 0.1 ether;

    function enter() public payable {
        require(msg.value == ticketPrice);
        players.push(msg.sender);
    }

    function random() private view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp, block.difficulty, players.length)));
    }

    function pickWinner() public {
        require(players.length >= 3);
        uint256 index = random() % players.length;
        address winner = players[index];
        delete players;
        winner.transfer(address(this).balance);
    }
}
